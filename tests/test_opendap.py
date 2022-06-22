"""Unit tests for OPeNDAP functionality."""

import pytest
from coast import OpendapInfo
from coast.data.opendap import CASTGC


@pytest.fixture(name="opendap_info")
def opendap_info_fixture(mocker) -> OpendapInfo:
    """Return an OPeNDAP accessor with mocked parameters."""
    return OpendapInfo(mocker.sentinel.url, session=mocker.sentinel.session)


def test_get_store(opendap_info, mocker) -> None:
    """Test that a store is instantiated with the correct parameters."""
    open_url = mocker.patch("coast.data.opendap.open_url", return_value=mocker.sentinel.dataset)
    pydap_data_store = mocker.patch("coast.data.opendap.PydapDataStore", return_value=mocker.sentinel.returned_store)

    store = opendap_info.get_store()

    open_url.assert_called_with(mocker.sentinel.url, session=mocker.sentinel.session)
    pydap_data_store.assert_called_with(mocker.sentinel.dataset)
    assert store == mocker.sentinel.returned_store


@pytest.mark.parametrize(["chunking"], [(True,), (False,)])
def test_open_dataset(opendap_info, mocker, chunking):
    """Test that a dataset is initialised with the correct parameters."""
    get_store = mocker.patch("coast.data.opendap.OpendapInfo.get_store", return_value=mocker.sentinel.store)
    dataset = mocker.Mock()
    dataset.__enter__ = mocker.Mock()  # Requirement of context manager
    dataset.__exit__ = mocker.Mock()  # Requirement of context manager
    open_dataset = mocker.patch("coast.data.opendap.open_dataset", return_value=dataset)
    chunks = mocker.Mock() if chunking else None

    dataset = opendap_info.open_dataset(chunks=chunks)

    assert dataset == dataset
    get_store.assert_called_once_with()
    open_dataset.assert_called_with(mocker.sentinel.store, chunks=chunks)


def test_from_cas(mocker):
    """Test that an OPeNDAP accessor is instantiated with the correct parameters."""
    mocker.sentinel.session = mocker.Mock()
    mocker.sentinel.session.cookies.get_dict.return_value = {"CASTGC": mocker.sentinel.castgc}

    setup_session = mocker.patch("coast.data.opendap.setup_session", return_value=mocker.sentinel.session)

    opendap_info = OpendapInfo.from_cas(
        mocker.sentinel.url, mocker.sentinel.cas_url, mocker.sentinel.username, mocker.sentinel.password
    )

    assert isinstance(opendap_info, OpendapInfo)
    assert opendap_info.url == mocker.sentinel.url
    assert opendap_info.session == mocker.sentinel.session
    assert setup_session.called_with(mocker.sentinel.cas_url, mocker.sentinel.username, mocker.sentinel.password)
    assert mocker.sentinel.session.cookies.set.called_with(CASTGC, mocker.sentinel.castgc)
    assert opendap_info.session == mocker.sentinel.session
