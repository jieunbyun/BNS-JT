import pytest
from BNS_JT import betasumrat

@pytest.fixture()
def ex1():
    a1, b1 = 2.50, 3.75
    a2, b2 = 1.25, 4.06
    return a1, b1, a2, b2

def test_betasumrat1(ex1):
    a1, b1, a2, b2 = ex1
    dist = betasumrat.BetaSumRat()

    density1 = dist.pdf(0.50, a1, a2, b1, b2)
    density2 = dist.pdf(0.500001, a1, a2, b1, b2)
    density3 = dist.pdf(0.499999, a1, a2, b1, b2)

    cprob = dist.cdf(0.999999, a1, a2, b1, b2)

    assert density1 == pytest.approx(density2, abs=1.0e-3) # Is the distribution continuous at 1/2?
    assert density2 == pytest.approx(density3, rel=1.0e-3) # Is the distribution continuous at 1/2?
    assert cprob == pytest.approx(1.0, rel=1.0e-3) # Does the CDF converge to 1?