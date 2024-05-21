import pytest
from discovery.utils import misc


@pytest.mark.parametrize(
    "input_dict, expected_output",
    [
        (
            {"a": {"key1": 1, "key2": 2}, "b": {"key1": 10, "key2": 20}},
            {"a.key1": 1, "a.key2": 2, "b.key1": 10, "b.key2": 20},
        ),
        (
            {"a": {"key1": {"subkey1": 1}, "key2": 2}, "b": {"key1": 10, "key2": 20}},
            {"a.key1.subkey1": 1, "a.key2": 2, "b.key1": 10, "b.key2": 20},
        ),
        (
            {"a": 1, "b": 2, "c": {"key1": 10, "key2": {"subkey1": 20}}},
            {"a": 1, "b": 2, "c.key1": 10, "c.key2.subkey1": 20},
        ),
        ({}, {}),
        ({"a": {"key1": 1, "key2": 2}, "b": {}}, {"a.key1": 1, "a.key2": 2, "b": {}}),
        (
            {
                "a": {"key1": 1, "key2": 2},
                "b": {"key1": 10, "key2": {"subkey1": 20, "subkey2": 30}},
            },
            {
                "a.key1": 1,
                "a.key2": 2,
                "b.key1": 10,
                "b.key2.subkey1": 20,
                "b.key2.subkey2": 30,
            },
        ),
    ],
)
def test_flatten_dict(input_dict, expected_output):
    assert misc.flatten_dict(input_dict) == expected_output


if __name__ == "__main__":
    pytest.main()
