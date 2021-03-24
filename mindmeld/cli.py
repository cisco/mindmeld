#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import datetime
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
import warnings
from shutil import which

import click
import click_log
import distro
import requests
from tqdm import tqdm

# Loads augmentor and annotator registration helper methods implicitly. Unused in this file.
from . import augmentation  # noqa: F401 pylint: disable=W0611
from .augmentation import AugmentorFactory
from .auto_annotator import register_all_annotators
from . import markup, path
from ._util import blueprint
from ._version import current as __version__
from .components import Conversation, QuestionAnswerer
from .components._config import (
    get_augmentation_config,
    get_auto_annotator_config,
    get_language_config,
)
from .constants import BINARIES_URL, DUCKLING_VERSION, UNANNOTATE_ALL_RULE
from .converter import DialogflowConverter, RasaConverter
from .exceptions import KnowledgeBaseConnectionError, KnowledgeBaseError, MindMeldError
from .models.helpers import create_annotator
from .path import (
    MODEL_CACHE_PATH,
    QUERY_CACHE_PATH,
    QUERY_CACHE_TMP_PATH,
    get_generated_data_folder,
    get_dvc_local_remote_path,
)
from .resource_loader import ResourceLoader

logger = logging.getLogger(__name__)
click.disable_unicode_literals_warning = True

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "auto_envvar_prefix": "MM"}

# deprecation warning for python 3.5
if sys.version_info < (3, 6):
    deprecation_msg = (
        "DEPRECATION: Python 3.5 reached end of life on 13 Sept 2020. MindMeld will deprecate"
        " official support for Python 3.5 in the next release. Please consider migrating"
        " your application to Python 3.6 and above."
    )
    logger.warning(deprecation_msg)

DVC_INIT_ERROR_MESSAGE = "you are not inside of a DVC repository"
DVC_ADD_DOES_NOT_EXIST_MESSAGE = "does not exist"

DVC_INIT_HELP = "Run 'dvc init' to instantiate this project as a DVC repository"
DVC_ADD_DOES_NOT_EXIST_HELP = "The folder {dvc_add_path} does not exist"

DVC_COMMAND_HELP_MESSAGE = (
    "Options:"
    "\n\t--init\t\tInstantiate DVC within a repository"
    "\n\t--save\t\tSave built models using dvc"
    "\n\t--checkout HASH\tCheckout repo and models corresponding to git hash"
    "\n\t--destroy\tRemove all files associated with DVC from a directory"
    "\n\t--help\t\tShow this message and exit\n"
)


def _version_msg():
    """Returns the MindMeld version, location and Python powering it."""
    python_version = sys.version[:3]
    location = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    message = "MindMeld %(version)s from {} (Python {})"
    return message.format(location, python_version)


#
# App only Commands
#


@click.group()
def _app_cli(ctx):
    """Command line interface for MindMeld apps."""

    # configure logger settings for dependent libraries
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.ERROR)
    es_logger = logging.getLogger("elasticsearch")
    es_logger.setLevel(logging.ERROR)

    if ctx.obj is None:
        ctx.obj = {}


def _dvc_add_helper(filepath):
    """
    Returns True if successful, False otherwise along with helper message

    Args:
        filepath (str): path to file/folder to add to DVC

    Returns:
        (tuple) True if no errors, False + error string otherwise
    """
    p = subprocess.Popen(
        ["dvc", "add", filepath], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Get DVC error message from standard error
    _, error = p.communicate()
    error_string = error.decode("utf-8")

    if DVC_INIT_ERROR_MESSAGE in error_string:
        return False, DVC_INIT_HELP
    elif DVC_ADD_DOES_NOT_EXIST_MESSAGE in error_string:
        return False, DVC_ADD_DOES_NOT_EXIST_HELP.format(dvc_add_path=filepath)
    elif p.returncode != 0:
        return False, error_string
    else:
        return True, None


def _bash_helper(command_list):
    """
    Helper for running bash using subprocess and error handling

    Args:
        command_list (list): Bash command formatted as a list, no spaces in each element

    Returns:
        (tuple) True if no errors, False + error string otherwise
    """
    p = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, error = p.communicate()
    error_string = error.decode("utf-8")

    if p.returncode != 0:
        return False, error_string

    return True, None


@_app_cli.command("dvc", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "--init", is_flag=True, required=False, help="Instantiate DVC within a repository"
)
@click.option(
    "--save", is_flag=True, required=False, help="Save built models using dvc"
)
@click.option("--checkout", required=False, help="Instantiate DVC within a repository")
@click.option(
    "--help",
    "help_",
    is_flag=True,
    required=False,
    help="Print message showing available options",
)
@click.option(
    "--destroy",
    is_flag=True,
    required=False,
    help="Remove all files associated with dvc from directory",
)
def dvc(ctx, init, save, checkout, help_, destroy):
    app = ctx.obj.get("app")
    app_path = app.app_path

    # Ensure that DVC is installed
    if not which("dvc"):
        logger.error(
            "DVC is not installed. You can install DVC by running 'pip install dvc'."
        )
        return

    if init:
        success, error_string = _bash_helper(["dvc", "init", "--subdir"])
        if not success:
            logger.error("Error during initialization: %s", error_string)
            return

        # Set up a local remote
        local_remote_path = get_dvc_local_remote_path(app_path)

        success, error_string = _bash_helper(
            ["dvc", "remote", "add", "-d", "myremote", local_remote_path]
        )
        if not success:
            logger.error("Error during local remote set up: %s", error_string)
            return

        # Add DVC config file to staging
        success, error_string = _bash_helper(["git", "add", ".dvc/config"])
        if not success:
            logger.error("Error while adding dvc config file: %s", error_string)
            return

        logger.info(
            "Instantiated DVC repo and set up local remote in %s", local_remote_path
        )
        logger.info(
            "The newly generated dvc config file (.dvc/config) has been added to git staging"
        )
    elif save:
        generated_model_folder = get_generated_data_folder(app_path)

        success, error_string = _dvc_add_helper(generated_model_folder)
        if not success:
            logger.error("Error during saving: %s", error_string)
            return

        success, error_string = _bash_helper(["dvc", "push"])
        if not success:
            logger.error("Error during dvc push: %s", error_string)
            return

        success, error_string = _bash_helper(
            ["git", "add", "{}/.generated.dvc".format(app_path)]
        )
        if not success:
            logger.error("Error adding model dvc file: %s", error_string)
            return

        logger.info("Successfully added .generated model folder to dvc")
        logger.info(
            "The newly generated .dvc file (%s/.generated.dvc) has been added to git staging",
            app_path,
        )
    elif checkout:
        success, error_string = _bash_helper(["git", "checkout", checkout])
        if not success:
            logger.error("Error during git checkout: %s", error_string)
            return

        success, error_string = _bash_helper(["dvc", "pull"])
        if not success:
            logger.error("Error during dvc checkout: %s", error_string)
            return

        logger.info(
            "Successfully checked out models corresponding to hash %s", checkout
        )
    elif destroy:
        logger.info(
            "This command must be run in the directory containing the .dvc/ folder. "
            "It will remove all files associated with dvc from the directory."
        )
        input("Press any key to continue:")

        # dvc destroy with -f flag always throws a benign error message so we don't handle
        _bash_helper(["dvc", "destroy", "-f"])
    elif help_:
        logger.info(DVC_COMMAND_HELP_MESSAGE)
    else:
        logger.error("No option provided, see options below.")
        logger.info(DVC_COMMAND_HELP_MESSAGE)


@_app_cli.command("run", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option("-P", "--port", type=int, default=7150)
@click.option(
    "--no-debug", is_flag=True, help="starts the service with debug mode turned off"
)
@click.option(
    "-r",
    "--reloader",
    is_flag=True,
    help="starts the service with the reloader enabled",
)
def run_server(ctx, port, no_debug, reloader):
    """Starts the MindMeld service."""
    app = ctx.obj.get("app")
    if app is None:
        raise ValueError(
            "No app was given. Run 'python app.py run' from your app folder."
        )

    # make sure num parser is running
    ctx.invoke(num_parser, start=True)

    app.run(
        port=port,
        debug=not no_debug,
        host="0.0.0.0",
        threaded=True,
        use_reloader=reloader,
    )


@_app_cli.command("converse", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option("--context", help="JSON object to be used as the context")
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print the full metrics instead of just accuracy.",
)
def converse(ctx, context, verbose):
    """
    Starts a conversation with the app.
    When the verbose flag is set to true, the confidences are included
    in the request objects passed to the intents
    """

    try:
        app = ctx.obj.get("app")
        if isinstance(context, str):
            context = json.loads(context)
        if app is None:
            raise ValueError(
                "No app was given. Run 'python app.py converse' from your app"
                " folder."
            )

        # make sure num parser is running
        ctx.invoke(num_parser, start=True)

        if app.async_mode:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(_converse_async(app, context))
            return

        convo = Conversation(app=app, context=context, verbose=verbose)

        while True:
            message = click.prompt("You")
            responses = convo.say(message)

            for index, response in enumerate(responses):
                prefix = "App: " if index == 0 else "...  "
                click.secho(prefix + response, fg="blue", bg="white")
    except MindMeldError as ex:
        logger.error(ex.message)
        ctx.exit(1)


async def _converse_async(app, context):
    convo = Conversation(app=app, context=context)
    while True:
        message = click.prompt("You")
        responses = await convo.say(message)

        for index, response in enumerate(responses):
            prefix = "App: " if index == 0 else "...  "
            click.secho(prefix + response, fg="blue", bg="white")


@_app_cli.command("build", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-i",
    "--incremental",
    is_flag=True,
    default=False,
    help="only build models with changed training data or configuration",
)
def build(ctx, incremental):
    """Builds the app with default config."""
    try:
        app = ctx.obj.get("app")
        if app is None:
            raise ValueError(
                "No app was given. Run 'python app.py build' from your app folder."
            )

        # make sure num parser is running
        ctx.invoke(num_parser, start=True)

        app.lazy_init()
        nlp = app.app_manager.nlp
        nlp.build(incremental=incremental)
        nlp.dump()
    except MindMeldError as ex:
        logger.error(ex.message)
        ctx.exit(1)
    except RuntimeError as ex:
        logger.error(ex)
        ctx.exit(1)


@_app_cli.command("evaluate", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print the full metrics instead of just accuracy.",
)
def evaluate(ctx, verbose):
    """Evaluates the app with default config."""
    try:
        app = ctx.obj.get("app")
        if app is None:
            raise ValueError(
                "No app was given. Run 'python app.py evaluate' from your app folder."
            )

        # make sure num parser is running
        ctx.invoke(num_parser, start=True)

        app.lazy_init()
        nlp = app.app_manager.nlp
        try:
            nlp.load()
        except MindMeldError:
            logger.error(
                "You must build the app before running evaluate. "
                "Try 'python app.py build'."
            )
            ctx.exit(1)
        nlp.evaluate(verbose)
    except MindMeldError as ex:
        logger.error(ex.message)
        ctx.exit(1)
    except RuntimeError as ex:
        logger.error(ex)
        ctx.exit(1)


@_app_cli.command("predict", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-o",
    "--output",
    required=False,
    help="Send output to file rather than standard out",
)
@click.option(
    "-c",
    "--confidence",
    is_flag=True,
    help="Show confidence scores for each prediction",
)
@click.option(
    "-D", "--no_domain", is_flag=True, help="Suppress predicted domain column"
)
@click.option(
    "-I", "--no_intent", is_flag=True, help="Suppress predicted intent column"
)
@click.option(
    "-E", "--no_entity", is_flag=True, help="Suppress predicted entity annotations"
)
@click.option(
    "-R", "--no_role", is_flag=True, help="Suppress predicted role annotations"
)
@click.option(
    "-G", "--no_group", is_flag=True, help="Suppress predicted group annotations"
)
@click.argument("input_file", envvar="INPUT", metavar="INPUT", required=True)
def predict(
    ctx,
    input_file,
    output,
    confidence,
    no_domain,
    no_intent,
    no_entity,
    no_role,
    no_group,
):
    """Runs predictions on a given query file"""
    app = ctx.obj.get("app")
    if app is None:
        raise ValueError(
            "No app was given. Run 'python app.py predict' from your app folder."
        )

    ctx.invoke(num_parser, start=True)

    app.lazy_init()
    nlp = app.app_manager.nlp
    try:
        nlp.load()
    except MindMeldError:
        logger.error(
            "You must build the app before running predict. "
            "Try 'python app.py build'."
        )
        ctx.exit(1)

    markup.bootstrap_query_file(
        input_file,
        output,
        nlp,
        confidence=confidence,
        no_domain=no_domain,
        no_intent=no_intent,
        no_entity=no_entity,
        no_role=no_role,
        no_group=no_group,
    )


@_app_cli.command("clean", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option(
    "-q", "--query-cache", is_flag=True, required=False, help="Clean only query cache"
)
@click.option(
    "-m", "--model-cache", is_flag=True, required=False, help="Clean only model cache"
)
@click.option(
    "-d",
    "--days",
    type=int,
    default=7,
    help="Clear model cache older than the specified days",
)
def clean(ctx, query_cache, model_cache, days):
    """Deletes all built data, undoing `build`."""
    app = ctx.obj.get("app")
    if app is None:
        raise ValueError(
            "No app was given. Run 'python app.py clean' from your app folder."
        )
    if query_cache:
        try:
            main_cache_location = QUERY_CACHE_PATH.format(app_path=app.app_path)
            tmp_cache_location = QUERY_CACHE_TMP_PATH.format(app_path=app.app_path)

            if os.path.exists(main_cache_location):
                os.remove(main_cache_location)

            if os.path.exists(tmp_cache_location):
                os.remove(tmp_cache_location)

            logger.info("Query cache deleted")
        except FileNotFoundError:
            logger.info("No query cache to delete")
        return

    if model_cache:
        model_cache_path = MODEL_CACHE_PATH.format(app_path=app.app_path)

        if not os.path.exists(model_cache_path):
            logger.warning("Model cache directory doesn't exist")
            return

        if days:
            for ts_folder in os.listdir(model_cache_path):
                full_path = os.path.join(model_cache_path, ts_folder)

                if not os.path.isdir(full_path):
                    logger.warning(
                        "Expected timestamped folder. Ignoring the file %s.", full_path
                    )
                    continue

                try:
                    current_ts = datetime.datetime.fromtimestamp(time.time())
                    folder_ts = datetime.datetime.strptime(
                        ts_folder, markup.TIME_FORMAT
                    )
                    diff_days = current_ts - folder_ts
                    if diff_days.days > days:
                        shutil.rmtree(full_path)
                        logger.info("Removed cached ts folder: %s", full_path)
                except ValueError:
                    logger.warning(
                        "Folder %s is not named as a proper timestamp. Ignoring it.",
                        full_path,
                    )
        else:
            try:
                shutil.rmtree(model_cache_path)
                logger.info("Model cache data deleted")
            except FileNotFoundError:
                logger.info("No model cache to delete")
        return

    gen_path = path.get_generated_data_folder(app.app_path)
    try:
        shutil.rmtree(gen_path)
        logger.info("Generated data deleted")
    except FileNotFoundError:
        logger.info("No generated data to delete")


#
# Shared commands
#


@click.group()
def shared_cli():
    """Commands for MindMeld module and apps"""
    pass


@shared_cli.command("load-kb", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option("-n", "--es-host", required=False, help="The ElasticSearch hostname.")
@click.argument("app_namespace", required=True)
@click.argument("index_name", required=True)
@click.argument("data_file", required=True)
@click.option(
    "--app-path",
    required=False,
    default=None,
    help="Needed to access app config to generating embeddings.",
)
def load_index(ctx, es_host, app_namespace, index_name, data_file, app_path):
    """Loads data into a question answerer index."""

    try:
        QuestionAnswerer.load_kb(
            app_namespace,
            index_name,
            data_file,
            es_host,
            app_path=app_path,
        )
    except (KnowledgeBaseConnectionError, KnowledgeBaseError) as ex:
        logger.error(ex.message)
        ctx.exit(1)


def _find_duckling_os_executable():
    """Returns the correct duckling path for this OS."""
    os_platform_name = "-".join(
        distro.linux_distribution(full_distribution_name=False)
    ).lower()
    for os_key in path.DUCKLING_OS_MAPPINGS:
        if os_key in os_platform_name:
            return path.DUCKLING_OS_MAPPINGS[os_key]


@shared_cli.command("num-parse", context_settings=CONTEXT_SETTINGS)
@click.option("--start/--stop", default=True, help="Start or stop numerical parser")
@click.option("-p", "--port", required=False, default="7151")
def num_parser(start, port):
    """Starts or stops the local numerical parser service."""
    if start:
        pid = _get_duckling_pid()

        if pid:
            # if duckling is already running, leave it be
            logger.info("Numerical parser running, PID %s", pid[0])
            return

        # We redirect all the output of starting the process to /dev/null and all errors
        # to stdout.
        exec_path = _find_duckling_os_executable()

        if not exec_path:
            logger.warning(
                "OS is incompatible with duckling executable. "
                "Use docker to install duckling."
            )
            return

        # Download the binary from the cloud if the binary does not already exist OR
        # the binary is out of date.
        if os.path.exists(exec_path):
            hash_digest = hashlib.md5(open(exec_path, "rb").read()).hexdigest()
            if hash_digest != path.DUCKLING_PATH_TO_MD5_MAPPINGS[exec_path]:
                os.remove(exec_path)

        if not os.path.exists(exec_path):
            url_components = [
                BINARIES_URL,
                "duckling",
                DUCKLING_VERSION,
                os.path.basename(exec_path),
            ]
            url = os.path.join(*url_components)
            logger.info(
                "Could not find %s binary file, downloading from %s", exec_path, url
            )
            r = requests.get(url, stream=True)

            # Total size in bytes.
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024

            with open(exec_path, "wb") as f:
                for data in tqdm(
                    r.iter_content(block_size),
                    total=math.ceil(total_size // block_size),
                    unit="KB",
                    unit_scale=True,
                ):
                    f.write(data)
                    f.flush()

        # make the file executable
        st = os.stat(exec_path)
        os.chmod(exec_path, st.st_mode | stat.S_IEXEC)

        # run duckling
        duckling_service = subprocess.Popen(
            [exec_path, "--port", port], stderr=subprocess.STDOUT
        )

        # duckling takes some time to start so sleep for a bit
        for _ in range(50):
            if duckling_service.pid:
                logger.info(
                    "Starting numerical parsing service, PID %s", duckling_service.pid
                )
                return
            time.sleep(0.1)
    else:
        for pid in _get_duckling_pid():
            os.kill(int(pid), signal.SIGKILL)
            logger.info("Stopping numerical parsing service, PID %s", pid)


def _get_duckling_pid():
    pid = []
    for line in os.popen("ps ax | grep duckling | grep -v grep"):
        pid.append(line.split()[0])
    return pid


@shared_cli.command("annotate", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--app-path",
    required=True,
    help="The application's path.",
)
@click.option(
    "--overwrite", is_flag=True, default=False, help="Overwrite existing annotations."
)
def annotate(app_path, overwrite):
    """Runs the annotation command of the Auto Annotator."""
    register_all_annotators()
    config = _get_auto_annotator_config(app_path=app_path, overwrite=overwrite)
    annotator = create_annotator(config)
    annotator.annotate()
    logger.info("Annotation Complete.")


@shared_cli.command("unannotate", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--app-path",
    required=True,
    help="The application's path.",
)
@click.option(
    "--unannotate_all",
    is_flag=True,
    default=False,
    help="Unnanotate all entities in app data.",
)
def unannotate(app_path, unannotate_all):
    """Runs the unannotation command of the Auto Annotator."""
    register_all_annotators()
    config = _get_auto_annotator_config(
        app_path=app_path, unannotate_all=unannotate_all
    )
    annotator = create_annotator(config)
    annotator.unannotate()
    logger.info("Annotation Removal Complete.")


def _get_auto_annotator_config(app_path, overwrite=False, unannotate_all=False):
    """ Gets the Annotator config from config.py. Overwrites params as needed."""
    config = get_auto_annotator_config(app_path=app_path)
    config["app_path"] = app_path
    config["language"], config["locale"] = get_language_config(app_path)
    if overwrite:
        config["overwrite"] = True
    if unannotate_all:
        config["unannotation_rules"] = UNANNOTATE_ALL_RULE
        config["unannotate_supported_entities_only"] = False
    return config


@shared_cli.command("augment", context_settings=CONTEXT_SETTINGS)
@click.option(
    "--app-path",
    required=True,
    help="The application's path.",
)
@click.option(
    "--language",
    help="Augmentation language code. Follows ISO 639-1 format.",
)
def augment(app_path, language):
    """Runs the data augmentation command."""
    config = get_augmentation_config(app_path=app_path)
    language = language or get_language_config(app_path=app_path)[0]
    resource_loader = ResourceLoader.create_resource_loader(app_path)
    augmentor = AugmentorFactory(
        config=config,
        language=language,
        resource_loader=resource_loader,
    ).create_augmentor()
    augmentor.augment()
    logger.info("Augmentation Complete.")


#
# Module only Commands
#


@click.group()
def module_cli():
    """Commands for MindMeld module only"""
    pass


@module_cli.command("blueprint", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option("-n", "--es-host")
@click.option("--skip-kb", is_flag=True, help="Skip setting up the knowledge base")
@click.argument("blueprint_name", required=True)
@click.argument("app_path", required=False)
def setup_blueprint(ctx, es_host, skip_kb, blueprint_name, app_path):
    """Sets up a blueprint application."""
    try:
        blueprint(blueprint_name, app_path, es_host=es_host, skip_kb=skip_kb)
    except ValueError as ex:
        logger.error(ex)
        ctx.exit(1)
    except (KnowledgeBaseConnectionError, KnowledgeBaseError) as ex:
        logger.error(ex.message)
        ctx.exit(1)


@module_cli.command("convert", context_settings=CONTEXT_SETTINGS)
@click.pass_context
@click.option("-d", "--df", is_flag=True, help="Convert a Dialogflow project")
@click.option("-r", "--rs", is_flag=True, help="Convert a Rasa project")
@click.argument("project_path", required=True, type=click.Path(exists=True))
@click.argument("mindmeld_path", required=False)
def convert(ctx, df, rs, project_path, mindmeld_path=None):
    """Converts a Rasa or DialogueFlow project to a MindMeld project"""
    if df:
        framework = "Dialogflow"
    elif rs:
        framework = "Rasa"
    else:
        logger.warning("Please specify the project's platform Rasa/Dialogflow.")
        ctx.exit(1)

    try:
        project_path = os.path.abspath(project_path)
        mindmeld_path = os.path.abspath(mindmeld_path or "converted_app")
        converter_cls = {"Rasa": RasaConverter, "Dialogflow": DialogflowConverter}
        converter_cls = converter_cls[framework]
        converter = converter_cls(project_path, mindmeld_path)
        converter.convert_project()
        msg = (
            "Successfully converted {framework} project at {project_path} to"
            " MindMeld project at {mindmeld_path}."
        )
        msg = msg.format(
            framework=framework, project_path=project_path, mindmeld_path=mindmeld_path
        )
        logger.info(msg)
    except IOError as e:
        logger.error(e)
        ctx.exit(1)


#
# Command collections
#


@click.command(
    cls=click.CommandCollection,
    context_settings=CONTEXT_SETTINGS,
    sources=[module_cli, shared_cli],
)
@click.version_option(__version__, "-V", "--version", message=_version_msg())
@click.pass_context
@click_log.simple_verbosity_option()
@click_log.init(__package__)
def cli(ctx):
    """Command line interface for MindMeld."""

    # configure logger settings for dependent libraries
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.ERROR)
    es_logger = logging.getLogger("elasticsearch")
    es_logger.setLevel(logging.ERROR)
    warnings.filterwarnings(
        "module", category=DeprecationWarning, module="sklearn.preprocessing.label"
    )
    if ctx.obj is None:
        ctx.obj = {}


@click.command(
    cls=click.CommandCollection,
    context_settings=CONTEXT_SETTINGS,
    sources=[_app_cli, shared_cli],
)
@click.version_option(__version__, "-V", "--version", message=_version_msg())
@click.pass_context
@click_log.simple_verbosity_option()
@click_log.init(__package__)
def app_cli(ctx):
    """Command line interface for MindMeld apps."""

    # configure logger settings for dependent libraries
    urllib3_logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.ERROR)
    es_logger = logging.getLogger("elasticsearch")
    es_logger.setLevel(logging.ERROR)
    warnings.filterwarnings(
        "module", category=DeprecationWarning, module="sklearn.preprocessing.label"
    )

    if ctx.obj is None:
        ctx.obj = {}


if __name__ == "__main__":
    cli({})
