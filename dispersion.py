"""Birefringence dispersion Δn(λ) of common nematic liquid crystals.

Extended (three-coefficient) Cauchy model for each index,

    n(λ) = A + B/λ² + C/λ⁴        (λ in µm),

so Δn(λ) = n_e − n_o = ΔA + ΔB/λ² + ΔC/λ⁴.

Coefficients are taken from the refractiveindex.info database
(github.com/polyanskiy/refractiveindex.info-database,
database/data/other/liquid crystals/<LC>/nk/*.yml, "formula 5"), which
transcribes the fits published in the papers cited below.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cauchy:
    A_e: float
    B_e: float
    C_e: float
    A_o: float
    B_o: float
    C_o: float
    valid_nm: tuple = (None, None)   # wavelength range of the original fit
    temperature: str = "unknown"
    reference: str = ""
    doi: str = ""

    @staticmethod
    def _n(lam_um, A, B, C):
        lam_um = np.asarray(lam_um, dtype=float)
        return A + B / lam_um**2 + C / lam_um**4

    def n_e(self, lam_um):
        return self._n(lam_um, self.A_e, self.B_e, self.C_e)

    def n_o(self, lam_um):
        return self._n(lam_um, self.A_o, self.B_o, self.C_o)

    def delta_n(self, lam_um):
        return self.n_e(lam_um) - self.n_o(lam_um)


LI_2005 = ("J. Li, C.-H. Wen, S. Gauza, R. Lu, S.-T. Wu (2005), "
           "Refractive indices of liquid crystals for display applications, "
           "J. Display Technol. 1, 51–61")
LI_2005_DOI = "10.1109/JDT.2005.853357"

TKACHENKO_2006 = ("V. Tkachenko, G. Abbate, A. Marino, F. Vita, M. Giocondo, "
                  "A. Mazzulla, F. Ciuchi, L. De Stefano (2006), Nematic liquid "
                  "crystal optical dispersion in the visible-near infrared range, "
                  "Mol. Cryst. Liq. Cryst. 454, 263–271")
TKACHENKO_2006_DOI = "10.1080/15421400600655816"

# material -> {dataset label -> Cauchy}
MATERIALS = {
    "E7": {
        "Li et al. 2005 (25 °C)": Cauchy(
            1.6933, 0.0078, 0.0028, 1.4990, 0.0072, 0.0003,
            valid_nm=(450, 656), temperature="25 °C",
            reference=LI_2005, doi=LI_2005_DOI),
        "Tkachenko et al. 2006": Cauchy(
            1.67798, 0.01696, 0.00127, 1.49669, 0.00785, 0.00026,
            valid_nm=(532, 1700), temperature="not stated in database entry",
            reference=TKACHENKO_2006, doi=TKACHENKO_2006_DOI),
    },
    "5CB": {
        "Li et al. 2005 (25 °C)": Cauchy(
            1.6708, 0.0081, 0.0024, 1.5139, 0.0052, 0.0008,
            valid_nm=(450, 656), temperature="25 °C",
            reference=LI_2005, doi=LI_2005_DOI),
        "Tkachenko et al. 2006": Cauchy(
            1.65535, 0.01355, 0.00153, 1.50849, 0.00774, 0.00040,
            valid_nm=(532, 1700), temperature="not stated in database entry",
            reference=TKACHENKO_2006, doi=TKACHENKO_2006_DOI),
    },
}

CUSTOM = "Custom (Cauchy)"
