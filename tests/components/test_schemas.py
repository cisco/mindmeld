import pytest

from mindmeld.components import schemas


@pytest.mark.parametrize('value, expected', (
        # str int in seconds
        ('1600000000', 1600000000000),
        # str int in ms
        ('1600000000000', 1600000000000),
        # str float in seconds
        ('1600000000.000', 1600000000000),
        # str float in seconds (preserve ms)
        ('1600000000.123', 1600000000123),
        # str float in seconds (preserve ms, round up)
        ('1600000000.1239', 1600000000124),
        # str float in ms
        ('1600000000000.100', 1600000000000),
        # str float in ms (round up)
        ('1600000000000.900', 1600000000001),
        # int in seconds
        (1600000000, 1600000000000),
        # int in ms
        (1600000000000, 1600000000000),
        # float in seconds
        (1600000000.000, 1600000000000),
        # float in seconds (preserve ms)
        (1600000000.123, 1600000000123),
        # float in seconds (preserve ms, round up)
        (1600000000.1239, 1600000000124),
        # float in ms
        (1600000000000.100, 1600000000000),
        # float in ms (round up)
        (1600000000000.900, 1600000000001),
    )
)
def test_validate_timestamp(value, expected):
    assert schemas.validate_timestamp(value) == expected
