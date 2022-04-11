.. _avail_opas:

Available opacity species
=========================

Line absorbers
______________

Please see the `Installation <installation.html>`_ section for how to
obtain and use the opacities listed below. For adding more opacity species not listed here,
please see `Adding opacities <opa_add.html>`_, among them how to plug-and-play install the Exomol opacities calculated
in the pRT format, available from the `Exomol website <http://www.exomol.com/data/data-types/opacity/>`_.

Default line absorbers, low resolution mode (``"c-k"``, :math:`\lambda/\Delta\lambda=1000`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**These are the default line absorbers contained in the input_data folder you download during the** `installation <installation.html>`_.

In low resolution mode (``"c-k"``), most of the molecular opacities are calculated considering
only the main isotopologue. This is different only for CO and TiO, where the contribution of all isotopologues is
considered. For CO because the secondary isotopes of carbon, for example :math:`\rm ^{13}C`, are quite abundant
when compared to the main isotope, that is :math:`\rm ^{12}C/^{13}C\sim 100`, and because CO has very strong and
sparse lines. Not including these lines therefore has a noticeable effect already at low resolution. For TiO all
isotopologues are included because the relative ratios between the Ti isotopes are quite large. Apart from these
two species, the main isotopologue treatment compared very well to codes including all isotopologues, at this low
resolution, see `Baudino et al. (2017) <http://adsabs.harvard.edu/abs/2017ApJ...850..150B>`_.

.. important::
   Please cite the reference mentioned in the description (click the link) when making use of a line species listed below.

.. list-table::
   :widths: 10 10 80
   :header-rows: 1

   * - Species name
     - Required in mass fraction dictionary
     - Reference
   * - AlH
     - AlH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Yurchenko+18 <https://doi.org/10.1093/mnras/sty1524>`_
   * - AlO
     - AlO
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Patrascu+15 <http://dx.doi.org/10.1093/mnras/stv507>`_
   * - C2H2
     - C2H2
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Chubb+20 <https://doi.org/10.1093/mnras/staa229>`_
   * - C2H4
     - C2H4
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Mant+18 <https://doi.org/10.1093/mnras/sty1239>`_
   * - CH4
     - CH4
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Yurchenko+17 <https://doi.org/10.1051/0004-6361/201731026>`_
   * - CO2
     - CO2
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Yurchenko+20 <https://doi.org/10.1093/mnras/staa1874>`_
   * - CO_all_iso_HITEMP
     - CO_all_iso_HITEMP
     - All isotopologues, HITEMP/Kurucz, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_12_HITEMP
     - CO_12_HITEMP
     - :math:`\rm ^{12}CO` isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_13_HITEMP
     - CO_13_HITEMP
     - :math:`\rm ^{13}CO` isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_all_iso_Chubb
     - CO_all_iso_Chubb
     - All isotopologues, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+15 <https://doi.org/10.1088/0067-0049/216/1/15>`_
   * - CO_13_Chubb
     - CO_13_Chubb
     - :math:`\rm ^{13}CO` isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+15 <https://doi.org/10.1088/0067-0049/216/1/15>`_
   * - CaH
     - CaH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Li+12 <http://dx.doi.org/10.1016/j.jqsrt.2011.09.010>`_
   * - CrH
     - CrH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Burrows+02 <http://dx.doi.org/10.1086/342242>`_
   * - FeH
     - FeH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Wende+10 <http://dx.doi.org/10.1051/0004-6361/201015220>`_
   * - H2O_HITEMP
     - H2O_HITEMP
     - Main isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2O_Exomol
     - H2O_Exomol
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Polyanski+18 <https://doi.org/10.1093/mnras/sty1877>`_
   * - H2S
     - H2S
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Azzam+16 <http://dx.doi.org/10.1093/mnras/stw1133>`_
   * - HCN
     - HCN
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Barber+14 <http://mnras.oxfordjournals.org/content/437/2/1828.abstract>`_
   * - K_allard
     - K_allard
     - Main isotopologue, VALD, Allard wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - K_burrows
     - K_burrows
     - Main isotopologue, VALD, `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
   * - K_lor_cut
     - K_lor_cut
     - Main isotopologue, VALD, Lorentzian wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - MgH
     - MgH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gharib-Nezhad+13 <http://dx.doi.org/10.1093/mnras/stt510>`_
   * - MgO
     - MgO
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Tennyson+19 <https://doi.org/10.1093/mnras/stz912>`_
   * - NH3
     - NH3
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Coles+19 <https://doi.org/10.1093/mnras/stz2778>`_
   * - NaH
     - NaH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Rivlin+15 <http://dx.doi.org/10.1093/mnras/stv979>`_
   * - Na_allard
     - Na_allard
     - Main isotopologue, VALD, `new Allard wings <https://ui.adsabs.harvard.edu/abs/2019yCat..36280120A/abstract>`_, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - Na_burrows
     - Na_burrows
     - Main isotopologue, VALD, `Burrows wings <https://ui.adsabs.harvard.edu/abs/2003ApJ...583..985B/abstract>`_
   * - Na_lor_cut
     - Na_lor_cut
     - Main isotopologue, VALD, Lorentzian wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - O2
     - O2
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gordon+17 <https://doi.org/10.1016/j.jqsrt.2017.06.038>`_
   * - O3
     - O3
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - OH
     - OH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Brooke+16 <http://dx.doi.org/10.1016/j.jqsrt.2015.07.021>`_
   * - PH3
     - PH3
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Sousa-Silva+14 <http://dx.doi.org/10.1093/mnras/stu2246>`_
   * - SH
     - SH
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Gorman+19 <https://doi.org/10.1093/mnras/stz2517>`_
   * - SiO
     - SiO
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Barton+13 <https://doi.org/10.1093/mnras/stt1105>`_
   * - SiO2
     - SiO2
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `Owens+20 <http://dx.doi.org/10.1093/mnras/staa1287>`_
   * - TiO_all_Plez
     - TiO_all_Plez
     - All isotopologues, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_48_Plez
     - TiO_48_Plez
     - :math:`\rm ^{48}TiO` isotopologue, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_all_Exomol
     - TiO_all_Exomol
     - All isotopologues, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `McKemmish+19 <https://doi.org/10.1093/mnras/stz1818>`_
   * - TiO_48_Exomol
     - TiO_48_Exomol
     - :math:`\rm ^{48}TiO` isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `McKemmish+19 <https://doi.org/10.1093/mnras/stz1818>`_
   * - VO_Plez
     - VO_Plez
     - Main isotopologue, B. Plez,, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - VO
     - VO
     - Main isotopologue, `ExoMolOP <https://ui.adsabs.harvard.edu/abs/2021A%26A...646A..21C/abstract>`_, `McKemmish+16 <http://dx.doi.org/10.1093/mnras/stw1969>`_

Contributed atom and ion opacities:

.. list-table::
   :widths: 10 10 10 10 10
   :header-rows: 1

   * - Name
     - Mass frac.
     - Ref. line list / broad.
     - P (bar), T (K) range
     - Contributor
   * - Al
     - Al
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Al+
     - Al+
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Ca
     - Ca
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Ca+
     - Ca+
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Fe
     - Fe
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Fe+
     - Fe+
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Li
     - Li
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_    
   * - Mg
     - Mg
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Mg+
     - Mg+
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - O
     - O
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Si
     - Si
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Si+
     - Si+
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Ti
     - Ti
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Ti+
     - Ti+
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - V
     - V
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - V+
     - V+
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_

**Line absorbers, high resolution mode** (``"lbl"``, with :math:`\lambda/\Delta\lambda=10^6`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10 10 80
   :header-rows: 1

   * - Species name
     - Required in mass fraction dictionary
     - Description
   * - C2H2_main_iso
     - C2H2_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CH4_212
     - CH4_212
     - :math:`\rm CH_3D`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CH4_main_iso
     - CH4_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO2_main_iso
     - CO2_main_iso
     - Main isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_27
     - CO_27
     - :math:`\rm ^{12}C^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_28
     - CO_28
     - :math:`\rm ^{12}C^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_36
     - CO_36
     - :math:`\rm ^{13}C^{16}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_37
     - CO_37
     - :math:`\rm ^{13}C^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_38
     - CO_38
     - :math:`\rm ^{13}C^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_all_iso
     - CO_all_iso
     - All isotopologues, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - CO_main_iso
     - CO_main_iso
     - Main isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2O_162
     - H2O_162
     - :math:`\rm HDO`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2O_171
     - H2O_171
     - :math:`\rm H_2 \ ^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2O_172
     - H2O_172
     - :math:`\rm HD^{17}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2O_181
     - H2O_181
     - :math:`\rm H_2 \ ^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2O_182
     - H2O_182
     - :math:`\rm HD^{18}O`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2O_main_iso
     - H2O_main_iso
     - Main isotopologue, HITEMP, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2S_main_iso
     - H2S_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2_12
     - H2_12
     - :math:`\rm HD`, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - H2_main_iso
     - H2_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - HCN_main_iso
     - HCN_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - K
     - K
     - Main isotopologue, VALD, Allard wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - NH3_main_iso
     - NH3_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - Na
     - Na
     - Main isotopologue, VALD, Allard wings, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - O3_main_iso
     - O3_main_iso
     - Main isotopologue, HITRAN, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - PH3_main_iso
     - PH3_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - SiO_main_iso
     - SiO_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_all_iso
     - TiO_all_iso
     - All isotopologues, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_46_Plez
     - TiO_46_Plez
     - :math:`\rm \ ^{46}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_47_Plez
     - TiO_47_Plez
     - :math:`\rm \ ^{47}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_48_Plez
     - TiO_48_Plez
     - :math:`\rm \ ^{48}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_49_Plez
     - TiO_49_Plez
     - :math:`\rm \ ^{49}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_50_Plez
     - TiO_50_Plez
     - :math:`\rm \ ^{50}TiO`, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - TiO_46_Exomol_McKemmish
     - TiO_46_Exomol_McKemmish
     - :math:`\rm \ ^{46}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
   * - TiO_47_Exomol_McKemmish
     - TiO_47_Exomol_McKemmish
     - :math:`\rm \ ^{47}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
   * - TiO_48_Exomol_McKemmish
     - TiO_48_Exomol_McKemmish
     - :math:`\rm \ ^{48}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
   * - TiO_49_Exomol_McKemmish
     - TiO_49_Exomol_McKemmish
     - :math:`\rm \ ^{49}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
   * - TiO_50_Exomol_McKemmish
     - TiO_50_Exomol_McKemmish
     - :math:`\rm \ ^{50}TiO`, Exomol, `McKemmish et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2836M/abstract>`_
   * - VO
     - VO
     - Main isotopologue, B. Plez, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_
   * - FeH_main_iso
     - FeH_main_iso
     - Main isotopologue, Exomol, see references in `here <https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..67M/abstract>`_

Contributed atom and ion opacities, high resolution mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 10 10 10 10 10
   :header-rows: 1

   * - Name
     - Mass frac.
     - Ref. line list / broad.
     - P (bar), T (K) range
     - Contributor
   * - Al
     - Al
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - B
     - B
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Be
     - Be
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - C
     - C
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Ca
     - Ca
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - CaII
     - CaII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Cr
     - Cr
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Fe
     - Fe
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - FeII
     - FeII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Li
     - Li
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_    
   * - Mg
     - Mg
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - MgII
     - MgII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - N
     - N
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_       
   * - Si
     - Si
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Ti
     - Ti
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - V
     - V
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - VII
     - VII
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - Y
     - Y
     - `Kurucz <http://kurucz.harvard.edu>`_, :math:`\gamma_{\rm nat+VdW},\sigma_{\rm therm}`
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `K. Molaverdikhani <karan@mpia.de>`_
   * - VO_ExoMol_McKemmish
     - VO_ExoMol_McKemmish
     - `McKemmish et al. (2016) <https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1969>`_
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `S. de Regt <regt@strw.leidenuniv.nl>`_
   * - VO_ExoMol_Specific_Transitions
     - VO_ExoMol_Specific_Transitions
     - Most accurate transitions from `McKemmish et al. (2016) <https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1969>`_
     - :math:`10^{-6}`-:math:`10^{3}`, 80-4000
     - `S. de Regt <regt@strw.leidenuniv.nl>`_
       
Cloud opacities
_______________

.. list-table::
   :widths: 10 10 80
   :header-rows: 1
		 
   * - Species name
     - Required in mass fraction dictionary
     - Description
   * - Al2O3(c)_cm
     - Al2O3(c)
     - Crystalline, Mie scattering (spherical)
   * - Al2O3(c)_cd
     - Al2O3(c)
     - Crystalline, DHS (irregular shape)
   * - Fe(c)_am
     - Fe(c)
     - Amorphous, Mie scattering (spherical)
   * - Fe(c)_ad
     - Fe(c)
     - Amorphous, DHS (irregular shape)
   * - Fe(c)_cm
     - Fe(c)
     - Crystalline, Mie scattering (spherical)
   * - Fe(c)_cd
     - Fe(c)
     - Crystalline, DHS (irregular shape)
   * - H2O(c)_cm
     - H2O(c)
     - Crystalline, Mie scattering (spherical)
   * - H2O(c)_cd
     - H2O(c)
     - Crystalline, DHS (irregular shape)
   * - KCL(c)_cm
     - KCL(c)
     - Crystalline, Mie scattering (spherical)
   * - KCL(c)_cd
     - KCL(c)
     - Crystalline, DHS (irregular shape)
   * - Mg05Fe05SiO3(c)_am
     - Mg05Fe05SiO3(c)
     - Amorphous, Mie scattering (spherical)
   * - Mg05Fe05SiO3(c)_ad
     - Mg05Fe05SiO3(c)
     - Amorphous, DHS (irregular shape)
   * - Mg2SiO4(c)_am
     - Mg2SiO4(c)
     - Amorphous, Mie scattering (spherical)
   * - Mg2SiO4(c)_ad
     - Mg2SiO4(c)
     - Amorphous, DHS (irregular shape)
   * - Mg2SiO4(c)_cm
     - Mg2SiO4(c)
     - Crystalline, Mie scattering (spherical)
   * - Mg2SiO4(c)_cd
     - Mg2SiO4(c)
     - Crystalline, DHS (irregular shape)
   * - MgAl2O4(c)_cm
     - MgAl2O4(c)
     - Crystalline, Mie scattering (spherical)
   * - MgAl2O4(c)_cd
     - MgAl2O4(c)
     - Crystalline, DHS (irregular shape)
   * - MgFeSiO4(c)_am
     - MgFeSiO4(c)
     - Amorphous, Mie scattering (spherical)
   * - MgFeSiO4(c)_ad
     - MgFeSiO4(c)
     - Amorphous, DHS (irregular shape)
   * - MgSiO3(c)_am
     - MgSiO3(c)
     - Amorphous, Mie scattering (spherical)
   * - MgSiO3(c)_ad
     - MgSiO3(c)
     - Amorphous, DHS (irregular shape)
   * - MgSiO3(c)_cm
     - MgSiO3(c)
     - Crystalline, Mie scattering (spherical)
   * - MgSiO3(c)_cd
     - MgSiO3(c)
     - Crystalline, DHS (irregular shape)
   * - Na2S(c)_cm
     - Na2S(c)
     - Crystalline, Mie scattering (spherical)
   * - Na2S(c)_cd
     - Na2S(c)
     - Crystalline, DHS (irregular shape)
   * - SiC(c)_cm
     - SiC(c)
     - Crystalline, Mie scattering (spherical)
   * - SiC(c)_cd
     - SiC(c)
     - Crystalline, DHS (irregular shape)
   
		 
Rayleigh scatterers
___________________

.. list-table::
   :widths: 10 10
   :header-rows: 1
		 
   * - Species name
     - Required in mass fraction dictionary
   * - H2
     - H2
   * - He
     - He
   * - H2O
     - H2O
   * - CO2
     - CO2
   * - O2
     - O2
   * - N2
     - N2
   * - CO
     - CO
   * - CH4
     - CH4


Continuum opacity sources
_________________________

.. list-table::
   :widths: 10 10 80
   :header-rows: 1
		 
   * - Species name
     - Required in mass fraction dictionary
     - Descripton
   * - H2-H2
     - H2
     - Collision induced absorption (CIA)
   * - H2-He
     - H2, He
     - Collision induced absorption (CIA)
   * - N2-N2
     - N2
     - Collision induced absorption (CIA)
   * - O2-O2
     - O2
     - Collision induced absorption (CIA)
   * - N2-O2
     - N2, O2
     - Collision induced absorption (CIA)
   * - CO2-CO2
     - CO2
     - Collision induced absorption (CIA)
   * - H-
     - H, H-, e-
     - H- bound-free and free-free opacity
