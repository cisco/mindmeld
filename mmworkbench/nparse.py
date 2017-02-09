import click
import errno
import os
import signal
import subprocess
import time


def _get_mallard_pid():
    pid = []
    for line in os.popen("ps ax | grep mindmeld-mallard.jar | grep -v grep"):
        pid.append(line.split()[0])
    return pid


@click.command()
@click.option('--start/--stop', default=True, help='Start or stop numerical parser service')
def mallard(start):
    """Simple command that starts or stops Mallard, the numerical parser service."""
    if start:
        pid = _get_mallard_pid()

        if len(pid) > 0:
            # if mallard is already running, leave it be
            click.echo("Numerical parser running, PID {0:s}".format(pid[0]))
            exit(0)

        pwd = os.path.dirname(os.path.abspath(__file__))
        mallard_path = os.path.join(pwd, 'mindmeld-mallard.jar')
        try:
            mallard_service = subprocess.Popen(["java", "-jar", mallard_path])
            time.sleep(5)
            click.echo("Starting numerical parsing service, PID %s" % mallard_service.pid)
        except OSError as e:
            if e.errno != errno.ENOENT:
                click.echo("Java is not found; please verify that Java 8 is installed and in your"
                           " path variable.")
                exit(1)
            else:
                raise e
    else:
        for pid in _get_mallard_pid():
            os.kill(int(pid), signal.SIGKILL)
            click.echo("Stopping numerical parsing service, PID %s" % pid)


if __name__ == '__main__':
    mallard()
