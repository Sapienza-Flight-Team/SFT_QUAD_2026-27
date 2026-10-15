from sft_math.typing import Float

EARTH_RADIUS: Float = Float(6371008.8)  # Raggio della terra in metri
#! https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml
#! Riferita alle coordinate:
#!   Lat, Lon = 38.31528, -76.55056
#!   WMMHR-2025
#!   2025/06/22
MAGNETIC_DECLINATION: Float = Float(
    -10.74319
)  # Declinazione magnetica in radianti, da aggiornare in base alla posizione
