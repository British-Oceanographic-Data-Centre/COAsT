import pytest
from coast import Copernicus, Product


@pytest.fixture(name="copernicus")
def copernicus_fixture(mocker):
    mocker.sentinel.template = mocker.Mock()
    mocker.sentinel.template.format.return_value = mocker.sentinel.url
    return Copernicus(mocker.sentinel.username, mocker.sentinel.password, mocker.sentinel.database, cas_url=mocker.sentinel.cas_url, url_template=mocker.sentinel.template)


def test_get_url(copernicus, mocker):
    url = copernicus.get_url(mocker.sentinel.product)

    assert url == mocker.sentinel.url
    assert mocker.sentinel.template.format.called_with(mocker.sentinel.database, mocker.sentinel.product)


def test_from_copernicus(copernicus, mocker):
    from_cas = mocker.patch("coast.data.opendap.OpendapInfo.from_cas", return_value=mocker.sentinel.product)

    product = Product.from_copernicus(mocker.sentinel.product_id, copernicus)

    assert product == mocker.sentinel.product
    from_cas.assert_called_with(mocker.sentinel.url, mocker.sentinel.cas_url, mocker.sentinel.username, mocker.sentinel.password)


def test_get_product(copernicus, mocker):
    from_copernicus = mocker.patch("coast.data.copernicus.Product.from_copernicus", return_value=mocker.sentinel.product)

    product = copernicus.get_product(mocker.sentinel.product_id)

    assert product == mocker.sentinel.product
    from_copernicus.assert_called_once_with(mocker.sentinel.product_id, copernicus)
