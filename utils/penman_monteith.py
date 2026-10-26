import math
import torch

def penman_monteith_pet(
    tmax,
    tmin,
    wind10,
    srad_sw,
    rel_hum,
    dayl,
    elev,
    rs,
    albedo,
    hc,
    lat,
    doy,
    zm=2.0,      
    zh=2.0,      
    h_max=2.5,   
):
    """
    Generalized Penman-Monteith PET for variable crop height.

    Inputs
    ------
    tmax, tmin : daily maximum/minimum air temperature [deg C]
    wind10     : wind speed measured at 10 m [m s-1]
    srad_sw    : mean daily shortwave radiation [W m-2]
    rel_hum    : mean daily relative humidity measured at 2 m [%]
    dayl       : daylight duration [s]
    elev       : elevation [m]
    rs         : surface/canopy resistance [s m-1]
    albedo     : surface albedo [-]
    hc          : crop height [m]
    lat        : latitude [degrees]
    doy        : day of year [1-365]
    zm, zh     : wind / humidity measurement heights [m], fixed at 2.0
    h_max      : crop height is clamped to this value before computing
                 the aerodynamic resistance, so that (zm - d) stays
                 safely positive (d = 2/3 * hc grows with hc while zm is
                 fixed at 2 m). With zm=2.0, h_max=2.5 gives
                 zm - d = 2.0 - 1.667 = 0.333 m of margin.

    Returns
    -------
    pet : potential evapotranspiration [mm day-1]
    """

    tmean = (tmax + tmin) / 2.0


    es_tmax = 0.6108 * torch.exp(17.27 * tmax / (tmax + 237.3))
    es_tmin = 0.6108 * torch.exp(17.27 * tmin / (tmin + 237.3))
    es = (es_tmax + es_tmin) / 2.0

    ea = (rel_hum / 100.0) * es

    vpd = torch.clamp(es - ea, min=0.0)

    wind_measurement_height = 10.0
    log_term = math.log(67.8 * wind_measurement_height - 5.42)

    u2 = torch.clamp(wind10 * 4.87 / log_term, min=0.1)


    Rs = srad_sw * dayl / 1e6      

    Rns = (1.0 - albedo) * Rs

    Gsc = 0.0820                   

    phi = lat * torch.pi / 180.0

    doy_f = (
        doy.float()
        if torch.is_tensor(doy)
        else torch.as_tensor(doy, dtype=tmax.dtype, device=tmax.device)
    )


    dr = 1.0 + 0.033 * torch.cos(2.0 * torch.pi * doy_f / 365.0)


    decl = 0.409 * torch.sin(2.0 * torch.pi * doy_f / 365.0 - 1.39)


    cos_ws = -torch.tan(phi) * torch.tan(decl)
    cos_ws = torch.clamp(cos_ws, min=-1.0, max=1.0)
    ws = torch.arccos(cos_ws)

    Ra = (
        (24.0 * 60.0 / torch.pi)
        * Gsc
        * dr
        * (ws * torch.sin(phi) * torch.sin(decl) + torch.cos(phi) * torch.cos(decl) * torch.sin(ws))
    )

    Ra = torch.clamp(Ra, min=1e-3)


    Rso = (0.75 + 2e-5 * elev) * Ra
    Rso = torch.clamp(Rso, min=1e-6)

    relative_radiation = Rs / Rso
    relative_radiation = torch.clamp(
        relative_radiation,
        min=0.3,
        max=1.0,
    )

    tmax_K = tmax + 273.16
    tmin_K = tmin + 273.16

    sigma = 4.903e-9       

    cloud_factor = 1.35 * relative_radiation - 0.35
    

    Rnl = (
        sigma
        * ((tmax_K ** 4 + tmin_K ** 4) / 2.0)
        * (0.34 - 0.14 * torch.sqrt(torch.clamp(ea, min=1e-6)))
        * cloud_factor
    )

    Rn = Rns - Rnl

    G = torch.zeros_like(Rn)

    es_tmean = 0.6108 * torch.exp(17.27 * tmean / (tmean + 237.3))
    delta = 4098.0 * es_tmean / (tmean + 237.3) ** 2


    P = 101.3 * ((293.0 - 0.0065 * elev) / 293.0) ** 5.26


    gamma = 0.000665 * P

    hc_safe = torch.clamp(hc, min=1e-3, max=h_max)


    d = (2.0 / 3.0) * hc_safe


    zom = 0.123 * hc_safe


    zoh = 0.1 * zom


    k = 0.41

    zm_minus_d = torch.clamp(zm - d, min=1e-3)
    zh_minus_d = torch.clamp(zh - d, min=1e-3)

    log_m = torch.log(torch.clamp(zm_minus_d / zom, min=1e-6))
    log_h = torch.log(torch.clamp(zh_minus_d / zoh, min=1e-6))

    ra = (log_m * log_h) / (k ** 2 * torch.clamp(u2, min=0.1))
    ra = torch.clamp(ra, min=10.0, max=2000.0)

    lambda_ = 2.45      

    rho_cp = 1.241e-3

    numerator = (
        delta * (Rn - G) / lambda_
        + (rho_cp / lambda_) * vpd * (86400.0 / ra)
    )

    denominator = delta + gamma * (1.0 + rs / ra)

    pet = numerator / (denominator + 1e-6)

    pet = torch.clamp(pet, min=0.0)

    return pet