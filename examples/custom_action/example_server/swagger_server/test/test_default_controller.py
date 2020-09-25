# coding: utf-8

from __future__ import absolute_import

from flask import json
from six import BytesIO

from swagger_server.models.data import Data
from swagger_server.models.responder import Responder
from swagger_server.test import BaseTestCase


class TestDefaultController(BaseTestCase):
    """DefaultController integration test stubs"""

    def test_invoke_action(self):
        """Test case for invoke_action

        Invoke an action
        """
        data = Data()
        response = self.client.open(
            "/v2/action",
            method="POST",
            data=json.dumps(data),
            content_type="application/json",
        )
        self.assert200(response, "Response body is : " + response.data.decode("utf-8"))


if __name__ == "__main__":
    import unittest

    unittest.main()
