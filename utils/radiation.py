import torch
import math

def compute_ra(lat, doy):
    """
    lat : [B] degrees
    doy : [B, T]
    returns Ra [B, T] MJ/m²/day
    """

    Gsc = 0.0820  

    lat_rad = lat * math.pi / 180.0
    lat_rad = lat_rad.unsqueeze(-1) 

    dr = 1 + 0.033 * torch.cos(2 * math.pi * doy / 365.0)
    delta = 0.409 * torch.sin(2 * math.pi * doy / 365.0 - 1.39)


    x = -torch.tan(lat_rad) * torch.tan(delta)
    x = torch.clamp(x, -1.0, 1.0)
    omega_s = torch.acos(x)

    Ra = (24 * 60 / math.pi) * Gsc * dr * (
        omega_s * torch.sin(lat_rad) * torch.sin(delta) +
        torch.cos(lat_rad) * torch.cos(delta) * torch.sin(omega_s)
    )

    return Ra


def compute_daylength(lat, doy):
    """
    lat: [B] in degrees
    doy: [B, T]

    returns: daylength [B, T] in seconds/day
    """

    lat_rad = lat * math.pi / 180.0
    lat_rad = lat_rad.unsqueeze(-1)   

    delta = 0.409 * torch.sin(2 * math.pi * doy / 365.0 - 1.39)

    x = -torch.tan(lat_rad) * torch.tan(delta)
    x = torch.clamp(x, -1.0, 1.0)
    omega_s = torch.acos(x)    

    N = (24.0 / math.pi) * omega_s  

    dayl = N * 3600.0  
 
    return dayl