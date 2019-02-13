import os

project_dir = os.path.dirname(os.path.dirname(__file__))


param_dict = dict(
    mu_e_ave_g = 'mu_ave_g',
    m_g = 'mag_auto_g',
    m_r = 'mag_auto_r',
    m_i = 'mag_auto_i',
    r_e = 'flux_radius_ave_g',
    ell='ellipticity'
    
)


labels = dict(
    m_g = r'$m_g$',
    m_r = r'$m_r$',
    m_i = r'$m_i$',
    mu_e_ave_g = r'$\bar{\mu}_\mathrm{eff}(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_e_ave_r = r'$\bar{\mu}_\mathrm{eff}(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_e_ave_i = r'$\bar{\mu}_\mathrm{eff}(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    r_e = r'$r_\mathrm{eff}$ [arcsec]',
    ell = 'Ellipticity'
)
