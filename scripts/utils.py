import os

project_dir = os.path.dirname(os.path.dirname(__file__))

labels = dict(
    mag_g = r'$m_g$',
    mag_r = r'$m_r$',
    mag_i = r'$m_i$',
    mu_0_g = r'$\mu_0(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_0_r = r'$\mu_0(r)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_0_i = r'$\mu_0(i)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_e_ave_g = r'$\bar{\mu}_\mathrm{eff}(g)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_e_ave_r = r'$\bar{\mu}_\mathrm{eff}(r)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    mu_e_ave_i = r'$\bar{\mu}_\mathrm{eff}(i)\ \mathrm{\left[mag\ arcsec^{-2}\right]}$',
    r_e = r'$r_\mathrm{eff}$ [arcsec]',
    ell = 'Ellipticity', 
    n = r'$n$'
)


