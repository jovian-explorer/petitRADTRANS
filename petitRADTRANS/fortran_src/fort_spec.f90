!!$*****************************************************************************
!!$*****************************************************************************
!!$*****************************************************************************
!!$ fort_spec.f90: utility functions to calculate cloud opacities, optical
!!$ depths, spectra, and spectral contribution functions for the petitRADTRANS
!!$ radiative transfer package.
!!$
!!$ Copyright 2016-2018, Paul Molliere
!!$ Maintained by Paul Molliere, molliere@strw.leidenunivl.nl
!!$ Status: under development
!!$*****************************************************************************
!!$*****************************************************************************
!!$*****************************************************************************

!!$ Natural constants block

module constants_block
  implicit none
  double precision,parameter      :: AU = 1.49597871d13, R_sun = 6.955d10, R_jup=6.9911d9
  double precision,parameter      :: pi = 3.14159265359d0, sig=5.670372622593201d-5, c_l=2.99792458d10
  double precision,parameter      :: G = 6.674d-8, M_jup = 1.89813d30, deg = Pi/1.8d2
  double precision,parameter      :: kB=1.3806488d-16, hplanck=6.62606957d-27, amu = 1.66053892d-24
  double precision,parameter      :: sneep_ubachs_n = 25.47d18, L0 = 2.68676d19
end module constants_block


!!$ #########################################################################
!!$ #########################################################################
!!$ #########################################################################
!!$ #########################################################################

!!$ Subroutine to calculate tau with 2nd order accuracy
module fort_spec
    implicit none

    contains
        subroutine calc_tau_g_tot_ck(gravity,press,total_kappa,struc_len,freq_len,g_len,N_species,tau)

          use constants_block
          implicit none

          ! I/O
          integer, intent(in)                          :: struc_len, freq_len, g_len, N_species
          double precision, intent(in)                 :: total_kappa(g_len,freq_len,N_species,struc_len)
          double precision, intent(in)                 :: gravity, press(struc_len)
          double precision, intent(out)                :: tau(g_len,freq_len,N_species,struc_len)
          ! internal
          integer                                      :: i_struc, i_freq, i_g, i_spec
          double precision                             :: del_tau_lower_ord, &
               gamma_second(g_len,freq_len,N_species), f_second, kappa_i(g_len,freq_len,N_species), &
               kappa_im(g_len,freq_len,N_species), kappa_ip(g_len,freq_len,N_species)
          logical                                      :: second_order
          !~~~~~~~~~~~~~

          tau = 0d0
          second_order = .FALSE.

          if (second_order) then
             do i_struc = 2, struc_len
                if (i_struc == struc_len) then
                   tau(:,:,:,i_struc) = tau(:,:,:,i_struc-1) + &
                        (total_kappa(:,:,:,i_struc)+total_kappa(:,:,:,i_struc-1)) &
                        /2d0/gravity*(press(i_struc)-press(i_struc-1))
                else
                   f_second = (press(i_struc+1)-press(i_struc))/(press(i_struc)-press(i_struc-1))
                   kappa_i = total_kappa(:,:,:,i_struc)
                   kappa_im = total_kappa(:,:,:,i_struc-1)
                   kappa_ip = total_kappa(:,:,:,i_struc+1)
                   gamma_second = (kappa_ip-(1d0+f_second)*kappa_i+f_second*kappa_im) / &
                        (f_second*(1d0+f_second))
                   tau(:,:,:,i_struc) = tau(:,:,:,i_struc-1) + &
                        ((kappa_i+kappa_im)/2d0-gamma_second/6d0) &
                        /gravity*(press(i_struc)-press(i_struc-1))
                   do i_spec = 1, N_species
                      do i_freq = 1, freq_len
                         do i_g = 1, g_len
                            if (tau(i_g,i_freq,i_spec,i_struc) < tau(i_g,i_freq,i_spec,i_struc-1)) then
                               if (i_struc <= 2) then
                                  tau(i_g,i_freq,i_spec,i_struc) = &
                                       tau(i_g,i_freq,i_spec,i_struc-1)*1.01d0
                               else
                                  tau(i_g,i_freq,i_spec,i_struc) = &
                                       tau(i_g,i_freq,i_spec,i_struc-1) + &
                                       (tau(i_g,i_freq,i_spec,i_struc-1)- &
                                       tau(i_g,i_freq,i_spec,i_struc-2))*0.01d0
                               end if
                            end if
                            del_tau_lower_ord = (kappa_i(i_g,i_freq,i_spec)+ &
                                 kappa_im(i_g,i_freq,i_spec))/2d0/gravity* &
                                 (press(i_struc)-press(i_struc-1))
                            if ((tau(i_g,i_freq,i_spec,i_struc) - &
                                 tau(i_g,i_freq,i_spec,i_struc-1)) > del_tau_lower_ord) then
                               tau(i_g,i_freq,i_spec,i_struc) = &
                                    tau(i_g,i_freq,i_spec,i_struc-1) + del_tau_lower_ord
                            end if
                         end do
                      end do
                   end do
                end if
             end do
          else
             do i_struc = 2, struc_len
                tau(:,:,:,i_struc) = tau(:,:,:,i_struc-1) + &
                     (total_kappa(:,:,:,i_struc)+total_kappa(:,:,:,i_struc-1)) &
                     /2d0/gravity*(press(i_struc)-press(i_struc-1))
             end do
          end if

        end subroutine calc_tau_g_tot_ck

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        !!$ Subroutine to calculate tau_scat with 2nd order accuracy

        subroutine calc_tau_g_tot_ck_scat(gravity,press,total_kappa_in,do_scat_emis, &
             continuum_opa_scat_emis,struc_len,freq_len,g_len,tau,photon_destruction_prob)

          use constants_block
          implicit none

          ! I/O
          integer, parameter                           :: N_species = 1
          integer, intent(in)                          :: struc_len, freq_len, g_len
          double precision, intent(in)                 :: total_kappa_in(g_len,freq_len,N_species,struc_len)
          double precision, intent(in)                 :: gravity, press(struc_len)
          logical, intent(in)                          :: do_scat_emis
          double precision, intent(in)                 :: continuum_opa_scat_emis(freq_len,struc_len)
          double precision, intent(out)                :: tau(g_len,freq_len,N_species,struc_len), &
               photon_destruction_prob(g_len,freq_len,struc_len)
          ! internal
          integer                                      :: i_struc, i_freq, i_g, i_spec
          double precision                             :: del_tau_lower_ord, &
               gamma_second(g_len,freq_len,N_species), f_second, kappa_i(g_len,freq_len,N_species), &
               kappa_im(g_len,freq_len,N_species), kappa_ip(g_len,freq_len,N_species)
          double precision                             :: total_kappa(g_len,freq_len,N_species,struc_len)
          logical                                      :: second_order
          !~~~~~~~~~~~~~

          tau = 0d0
          second_order = .FALSE.

          total_kappa = total_kappa_in

          if (do_scat_emis) then
             do i_g = 1, g_len
                total_kappa(i_g,:,1,:) = total_kappa(i_g,:,1,:) + &
                     continuum_opa_scat_emis(:,:)
                photon_destruction_prob(i_g,:,:) = continuum_opa_scat_emis(:,:) / &
                     total_kappa(i_g,:,1,:)
             end do
             photon_destruction_prob = 1d0 - photon_destruction_prob
          else
             photon_destruction_prob = 1d0
          end if

          if (second_order) then
             do i_struc = 2, struc_len
                if (i_struc == struc_len) then
                   tau(:,:,:,i_struc) = tau(:,:,:,i_struc-1) + &
                        (total_kappa(:,:,:,i_struc)+total_kappa(:,:,:,i_struc-1)) &
                        /2d0/gravity*(press(i_struc)-press(i_struc-1))
                else
                   f_second = (press(i_struc+1)-press(i_struc))/(press(i_struc)-press(i_struc-1))
                   kappa_i = total_kappa(:,:,:,i_struc)
                   kappa_im = total_kappa(:,:,:,i_struc-1)
                   kappa_ip = total_kappa(:,:,:,i_struc+1)
                   gamma_second = (kappa_ip-(1d0+f_second)*kappa_i+f_second*kappa_im) / &
                        (f_second*(1d0+f_second))
                   tau(:,:,:,i_struc) = tau(:,:,:,i_struc-1) + &
                        ((kappa_i+kappa_im)/2d0-gamma_second/6d0) &
                        /gravity*(press(i_struc)-press(i_struc-1))
                   do i_spec = 1, N_species
                      do i_freq = 1, freq_len
                         do i_g = 1, g_len
                            if (tau(i_g,i_freq,i_spec,i_struc) < tau(i_g,i_freq,i_spec,i_struc-1)) then
                               if (i_struc <= 2) then
                                  tau(i_g,i_freq,i_spec,i_struc) = &
                                       tau(i_g,i_freq,i_spec,i_struc-1)*1.01d0
                               else
                                  tau(i_g,i_freq,i_spec,i_struc) = &
                                       tau(i_g,i_freq,i_spec,i_struc-1) + &
                                       (tau(i_g,i_freq,i_spec,i_struc-1)- &
                                       tau(i_g,i_freq,i_spec,i_struc-2))*0.01d0
                               end if
                            end if
                            del_tau_lower_ord = (kappa_i(i_g,i_freq,i_spec)+ &
                                 kappa_im(i_g,i_freq,i_spec))/2d0/gravity* &
                                 (press(i_struc)-press(i_struc-1))
                            if ((tau(i_g,i_freq,i_spec,i_struc) - &
                                 tau(i_g,i_freq,i_spec,i_struc-1)) > del_tau_lower_ord) then
                               tau(i_g,i_freq,i_spec,i_struc) = &
                                    tau(i_g,i_freq,i_spec,i_struc-1) + del_tau_lower_ord
                            end if
                         end do
                      end do
                   end do
                end if
             end do
          else
             do i_struc = 2, struc_len
                tau(:,:,:,i_struc) = tau(:,:,:,i_struc-1) + &
                     (total_kappa(:,:,:,i_struc)+total_kappa(:,:,:,i_struc-1)) &
                     /2d0/gravity*(press(i_struc)-press(i_struc-1))
             end do
          end if

        end subroutine calc_tau_g_tot_ck_scat

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        subroutine calc_kappa_rosseland(total_kappa, temp, w_gauss, border_freqs, &
             do_scat_emis, continuum_opa_scat_emis, &
             g_len, freq_len, struc_len, freq_len_p_1, kappa_rosse)

          implicit none

          integer,          intent(in)  :: g_len, freq_len, struc_len, freq_len_p_1
          double precision, intent(in)  :: total_kappa(g_len, freq_len, struc_len)
          double precision, intent(in)  :: border_freqs(freq_len_p_1)
          double precision, intent(in)  :: temp(struc_len), w_gauss(g_len)
          logical, intent(in)           :: do_scat_emis
          double precision, intent(in)  :: continuum_opa_scat_emis(freq_len,struc_len)
          double precision, intent(out) :: kappa_rosse(struc_len)

          double precision              :: total_kappa_use(g_len, freq_len, struc_len)

          integer                       :: i_struc, i_g

          if (do_scat_emis) then
             do i_g = 1, g_len
                total_kappa_use(i_g,:,:) = total_kappa(i_g,:,:) + continuum_opa_scat_emis
             end do
          else
             total_kappa_use = total_kappa
          end if

          do i_struc = 1, struc_len
             call calc_rosse_opa(total_kappa_use(:,:,i_struc), border_freqs, temp(i_struc), &
                  g_len, freq_len+1, &
                  kappa_rosse(i_struc), w_gauss)
          end do

        end subroutine calc_kappa_rosseland

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        subroutine calc_kappa_planck(total_kappa, temp, w_gauss, border_freqs, &
             do_scat_emis, continuum_opa_scat_emis, &
             g_len, freq_len, struc_len, freq_len_p_1, kappa_planck)

          implicit none

          integer,          intent(in)  :: g_len, freq_len, struc_len, freq_len_p_1
          double precision, intent(in)  :: total_kappa(g_len, freq_len, struc_len)
          double precision, intent(in)  :: border_freqs(freq_len_p_1)
          double precision, intent(in)  :: temp(struc_len), w_gauss(g_len)
          logical, intent(in)           :: do_scat_emis
          double precision, intent(in)  :: continuum_opa_scat_emis(freq_len,struc_len)
          double precision, intent(out) :: kappa_planck(struc_len)

          double precision              :: total_kappa_use(g_len, freq_len, struc_len)

          integer                       :: i_struc, i_g

          if (do_scat_emis) then
             do i_g = 1, g_len
                total_kappa_use(i_g,:,:) = total_kappa(i_g,:,:) + continuum_opa_scat_emis
             end do
          else
             total_kappa_use = total_kappa
          end if

          do i_struc = 1, struc_len
             call calc_planck_opa(total_kappa_use(:,:,i_struc), border_freqs, temp(i_struc), &
                  g_len, freq_len+1, &
                  kappa_planck(i_struc), w_gauss)
          end do

        end subroutine calc_kappa_planck

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        !!$ Subroutine to do the radiative transport, using the mean transmission method

        subroutine flux_ck(freq,tau,temp,mu,w_gauss_mu, &
             w_gauss,contribution,freq_len,struc_len,N_mu,g_len,N_species,flux,contr_em)

          use constants_block
          implicit none

          ! I/O
          integer, intent(in)                         :: freq_len, struc_len,g_len, N_species
          double precision, intent(in)                :: freq(freq_len)
          double precision, intent(in)                :: temp(struc_len) !, press(struc_len)
          double precision, intent(in)                :: tau(g_len,freq_len,N_species,struc_len)

          integer, intent(in)                         :: N_mu
          double precision, intent(in)                :: mu(N_mu) !, gravity
          double precision, intent(in)                :: w_gauss_mu(N_mu)
          double precision, intent(in)                :: w_gauss(g_len)
          logical, intent(in)                         :: contribution
          double precision, intent(out)               :: flux(freq_len)
          double precision, intent(out)               :: contr_em(struc_len,freq_len)

          ! Internal
          integer                                     :: i_mu,i_freq,i_str,i_spec
          double precision                            :: r(struc_len)
          double precision                            :: transm_mu(g_len,freq_len,N_species,struc_len), &
               mean_transm(freq_len,N_species,struc_len), transm_all(freq_len,struc_len), &
               transm_all_loc(struc_len), flux_mu(freq_len)

          flux = 0d0

          if (contribution) then
             contr_em = 0d0
          end if

          do i_mu = 1, N_mu

             ! will contain species' product of g-space integrated transmissions
             transm_all = 1d0
             ! Transmissions for a given incidence angle
             transm_mu = exp(-tau/mu(i_mu))
             ! Flux contribution from given mu-angle
             flux_mu = 0d0

             do i_str = 1, struc_len
                do i_spec = 1, N_species
                   do i_freq = 1, freq_len
                      ! Integrate transmission over g-space
                      mean_transm(i_freq,i_spec,i_str) = sum(transm_mu(:,i_freq,i_spec,i_str)*w_gauss)
                   end do
                end do
             end do

             ! Multiply transmissions of infdiv. species
             do i_spec = 1, N_species
                transm_all = transm_all*mean_transm(:,i_spec,:)
             end do

             ! Do the actual radiative transport
             do i_freq = 1, freq_len
                ! Get source function
                r = 0
                call planck_f(struc_len,temp,freq(i_freq),r)
                ! Spatial transmissions at given wavelength
                transm_all_loc = transm_all(i_freq,:)
                ! Calc Eq. 9 of manuscript (em_deriv.pdf)
                do i_str = 1, struc_len-1
                   flux_mu(i_freq) = flux_mu(i_freq)+ &
                        (r(i_str)+r(i_str+1))*(transm_all_loc(i_str)-transm_all_loc(i_str+1))/2d0
                   if (contribution) then
                      contr_em(i_str,i_freq) = contr_em(i_str,i_freq) &
                          + (r(i_str)+r(i_str+1)) * &
                           (transm_all_loc(i_str)-transm_all_loc(i_str+1)) &
                           *mu(i_mu)*w_gauss_mu(i_mu)
                   end if
                end do
                flux_mu(i_freq) = flux_mu(i_freq) + r(struc_len)*transm_all_loc(struc_len)
                if (contribution) then
                   contr_em(struc_len,i_freq) = contr_em(struc_len,i_freq) + 2d0*r(struc_len)* &
                        transm_all_loc(struc_len)*mu(i_mu)*w_gauss_mu(i_mu)
                end if
             end do
             ! angle integ, factor 1/2 needed for flux calc. from upward pointing intensity
             flux = flux + flux_mu/2d0*mu(i_mu)*w_gauss_mu(i_mu)

          end do
          ! Normalization
          flux = flux*4d0*pi

          if (contribution) then
             do i_freq = 1, freq_len
                contr_em(:,i_freq) = contr_em(:,i_freq)/sum(contr_em(:,i_freq))
             end do
          end if

        end subroutine flux_ck

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        !!$ Subroutine to calculate the Planck source function

        subroutine planck_f(struc_len,T,nu,B_nu)

          use constants_block
          implicit none
          integer                         :: struc_len
          double precision                :: T(struc_len),B_nu(struc_len), nu
          double precision                :: buffer

          !~~~~~~~~~~~~~

          B_nu = 0d0
          buffer = 2d0*hplanck*nu**3d0/c_l**2d0
          B_nu = buffer / (exp(hplanck*nu/kB/T)-1d0)

        end subroutine planck_f

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        !!$ Subroutine to calculate the transmission spectrum

        subroutine calc_transm_spec(total_kappa_in,temp,press,gravity,mmw,P0_bar,R_pl, &
             w_gauss,scat,continuum_opa_scat,var_grav,transm,freq_len,struc_len,g_len,N_species)

          use constants_block
          implicit none

          ! I/O
          integer, intent(in)                         :: freq_len, struc_len, g_len, N_species
          double precision, intent(in)                :: P0_bar, R_pl
          double precision, intent(in)                :: temp(struc_len), press(struc_len), mmw(struc_len)
          double precision, intent(in)                :: total_kappa_in(g_len,freq_len,N_species,struc_len)

          double precision, intent(in)                :: gravity
          double precision, intent(in)                :: w_gauss(g_len), continuum_opa_scat(freq_len,struc_len)
          logical, intent(in)                         :: scat !, contribution
          logical, intent(in)                         :: var_grav

          double precision, intent(out)               :: transm(freq_len) !, contr_tr(struc_len,freq_len)

          ! Internal
          double precision                            :: P0_cgs, rho(struc_len), radius(struc_len), &
                total_kappa(g_len,freq_len,N_species,struc_len)
          integer                                     :: i_str, i_freq, i_g, i_spec, j_str
          logical                                     :: rad_neg
          double precision                            :: alpha_t2(g_len,freq_len,N_species,struc_len-1)
          double precision                            :: t_graze(g_len,freq_len,N_species,struc_len), s_1, s_2, &
               t_graze_wlen_int(struc_len,freq_len), &
               alpha_t2_scat(freq_len,struc_len-1), t_graze_scat(freq_len,struc_len)

          total_kappa = total_kappa_in
          ! Some cloud opas can be < 0 sometimes, apparently.
          do i_str = 1, struc_len
             do i_spec = 1, N_species
                do i_freq = 1, freq_len
                   do i_g = 1, g_len
                      if (total_kappa(i_g,i_freq,i_spec,i_str) < 0d0) then
                         total_kappa(i_g,i_freq,i_spec,i_str) = 0d0
                      end if
                   end do
                end do
             end do
          end do

          transm = 0d0
          t_graze = 0d0
          t_graze_scat = 0d0

          ! Convert reference pressure to cgs
          P0_cgs = P0_bar*1d6
          ! Calculate density
          rho = mmw*amu*press/kB/temp
          ! Calculate planetary radius (in cm), assuming hydrostatic equilibrium
          call calc_radius(struc_len,press,gravity,rho,P0_cgs,R_pl,var_grav,radius)

          rad_neg = .FALSE.
          do i_str = struc_len, 1, -1
             if (radius(i_str) < 0d0) then
                rad_neg = .TRUE.
                radius(i_str) = radius(i_str+1)
             end if
          end do
          if (rad_neg) then
             write(*,*) 'pRT: negative radius corretion applied!'
          end if

          ! Calc. mean free paths across grazing distances
          do i_str = 1, struc_len-1
             alpha_t2(:,:,:,i_str) = (total_kappa(:,:,:,i_str)*rho(i_str)+total_kappa(:,:,:,i_str+1)*rho(i_str+1))
          end do

          if (scat) then
             do i_str = 1, struc_len-1
                alpha_t2_scat(:,i_str) = (continuum_opa_scat(:,i_str)*rho(i_str)+ &
                     continuum_opa_scat(:,i_str+1)*rho(i_str+1))
             end do
          end if

          ! Cacuclate grazing rays optical depths
          do i_str = 2, struc_len
             s_1 = sqrt(radius(1)**2d0-radius(i_str)**2d0)
             do j_str = 1, i_str-1
                if (j_str > 1) then
                   s_1 = s_2
                end if
                s_2 = sqrt(radius(j_str+1)**2d0-radius(i_str)**2d0)
                t_graze(:,:,:,i_str) = t_graze(:,:,:,i_str)+alpha_t2(:,:,:,j_str)*(s_1-s_2)
             end do
          end do
          if (scat) then
             do i_str = 2, struc_len
                s_1 = sqrt(radius(1)**2d0-radius(i_str)**2d0)
                do j_str = 1, i_str-1
                   if (j_str > 1) then
                      s_1 = s_2
                   end if
                   s_2 = sqrt(radius(j_str+1)**2d0-radius(i_str)**2d0)
                   t_graze_scat(:,i_str) = t_graze_scat(:,i_str)+alpha_t2_scat(:,j_str)*(s_1-s_2)
                end do
             end do
          end if

          ! Calculate transmissions, update tau array to store these
          t_graze = exp(-t_graze)
          if (scat) then
             t_graze_scat = exp(-t_graze_scat)
          end if

          t_graze_wlen_int = 1d0
          ! Wlen (in g-space) integrate transmissions
          do i_str = 2, struc_len ! i_str=1 t_grazes are 1 anyways
             do i_spec = 1, N_species
                do i_freq = 1, freq_len
                   t_graze_wlen_int(i_str,i_freq) = t_graze_wlen_int(i_str,i_freq)* &
                        sum(t_graze(:,i_freq,i_spec,i_str)*w_gauss)
                   if (scat .and. (i_spec == 1)) then
                      t_graze_wlen_int(i_str,i_freq) = t_graze_wlen_int(i_str,i_freq)* &
                           t_graze_scat(i_freq,i_str)
                   end if
                end do
             end do
          end do

          ! Get effective area fraction from transmission
          t_graze_wlen_int = 1d0-t_graze_wlen_int

          ! Caculate planets effectice area (leaving out pi, because we want the radius in the end)
          do i_freq = 1, freq_len
             do i_str = 2, struc_len
                transm(i_freq) = transm(i_freq)+(t_graze_wlen_int(i_str-1,i_freq)*radius(i_str-1)+ &
                     t_graze_wlen_int(i_str,i_freq)*radius(i_str))*(radius(i_str-1)-radius(i_str))
             end do
          end do
          ! Get radius
          transm = sqrt(transm+radius(struc_len)**2d0)

        !!$  if (contribution) then
        !!$     contr_tr = t_graze_wlen_int
        !!$  end if

        !!$  call calc_radius(struc_len,temp,press,gravity,mmw,rho,P0_cgs,R_pl,.FALSE.,radius)
        !!$  call calc_radius(struc_len,temp,press,gravity,mmw,rho,P0_cgs,R_pl,.TRUE., radius_var)
        !!$  open(unit=10,file='rad_test.dat')
        !!$  do i_str = 1, struc_len
        !!$     write(10,*) press(i_str)*1d-6, radius(i_str)/R_jup, radius_var(i_str)/R_jup
        !!$  end do
        !!$  close(10)

        end subroutine calc_transm_spec

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        !!$ Subroutine to calculate the radius from the pressure grid

        subroutine calc_radius(struc_len,press,gravity,rho,P0_cgs, &
             R_pl,var_grav,radius)

          implicit none
          ! I/O
          integer, intent(in)                         :: struc_len
          double precision, intent(in)                :: P0_cgs
          double precision, intent(in)                :: press(struc_len), &
               rho(struc_len)
          double precision, intent(in)                :: gravity, R_pl
          logical, intent(in)                         :: var_grav
          double precision, intent(out)               :: radius(struc_len)

          ! Internal
          integer                                     :: i_str
          double precision                            :: R0, inv_rho(struc_len)

          inv_rho = 1d0/rho

          radius = 0d0
          R0=0d0
          if (var_grav) then

             !write(*,*) '####################### VARIABLE GRAVITY'
             !write(*,*) '####################### VARIABLE GRAVITY'
             !write(*,*) '####################### VARIABLE GRAVITY'
             !write(*,*) '####################### VARIABLE GRAVITY'

             ! Calculate radius with vertically varying gravity, set up such that at P=P0, i.e. R=R_pl
             ! the planet has the predefined scalar gravity value
             do i_str = struc_len-1, 1, -1
                if ((press(i_str+1) > P0_cgs) .and. (press(i_str) <= P0_cgs)) then
                   if (i_str <= struc_len-2) then
                      R0 = radius(i_str+1) - integ_parab(press(i_str),press(i_str+1),press(i_str+2), &
                           inv_rho(i_str),inv_rho(i_str+1),inv_rho(i_str+2),P0_cgs,press(i_str+1))/gravity &
                           /R_pl**2d0
                   else
                      R0 = radius(i_str+1)-(1d0/rho(i_str)+1d0/rho(i_str+1))/(2d0*gravity)* &
                           (press(i_str+1)-P0_cgs)/R_pl**2d0
                   end if
                end if
                if (i_str <= struc_len-2) then
                   radius(i_str) = radius(i_str+1) - integ_parab(press(i_str),press(i_str+1),press(i_str+2), &
                        inv_rho(i_str),inv_rho(i_str+1),inv_rho(i_str+2),press(i_str),press(i_str+1))/gravity &
                        /R_pl**2d0
                else
                   radius(i_str) = radius(i_str+1)-(1d0/rho(i_str)+1d0/rho(i_str+1))/(2d0*gravity)* &
                        (press(i_str+1)-press(i_str))/R_pl**2d0
                end if
             end do
             R0 = 1d0/R_pl-R0
             radius = radius + R0
             radius = 1d0/radius

          else

             !write(*,*) '####################### CONSTANT GRAVITY'
             !write(*,*) '####################### CONSTANT GRAVITY'
             !write(*,*) '####################### CONSTANT GRAVITY'
             !write(*,*) '####################### CONSTANT GRAVITY'


             ! Calculate radius with vertically constant gravity
             do i_str = struc_len-1, 1, -1
                if ((press(i_str+1) > P0_cgs) .and. (press(i_str) <= P0_cgs)) then
                   if (i_str <= struc_len-2) then
                      R0 = radius(i_str+1) + integ_parab(press(i_str),press(i_str+1),press(i_str+2), &
                           inv_rho(i_str),inv_rho(i_str+1),inv_rho(i_str+2),P0_cgs,press(i_str+1))/gravity
                   else
                      R0 = radius(i_str+1)+(1d0/rho(i_str)+1d0/rho(i_str+1))/(2d0*gravity)* &
                           (press(i_str+1)-P0_cgs)
                   end if
                end if
                if (i_str <= struc_len-2) then
                   radius(i_str) = radius(i_str+1) + integ_parab(press(i_str),press(i_str+1),press(i_str+2), &
                        inv_rho(i_str),inv_rho(i_str+1),inv_rho(i_str+2),press(i_str),press(i_str+1))/gravity
                else
                   radius(i_str) = radius(i_str+1)+(1d0/rho(i_str)+1d0/rho(i_str+1))/(2d0*gravity)* &
                        (press(i_str+1)-press(i_str))
                end if
             end do

             R0 = R_pl-R0
             radius = radius + R0
        !!$     write(*,*) R0, P0_cgs, gravity, R_pl, press(20), rho(20)

          end if


            contains
                !!$ Function to calc higher order integ.
                function integ_parab(x,y,z,fx,fy,fz,a,b)

                  implicit none
                  ! I/O
                  double precision :: x,y,z,fx,fy,fz,a,b
                  double precision :: integ_parab
                  ! Internal
                  double precision :: c1,c2,c3

                  c3 = ((fz-fy)/(z-y)-(fz-fx)/(z-x))/(y-x)
                  c2 = (fz-fx)/(z-x)-c3*(z+x)
                  c1 = fx-c2*x-c3*x**2d0

                  integ_parab = c1*(b-a)+c2*(b**2d0-a**2d0)/2d0+c3*(b**3d0-a**3d0)/3d0

                end function integ_parab
        end subroutine calc_radius

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        !!$ Subroutine to add Rayleigh scattering

        subroutine add_rayleigh(spec,abund,lambda_angstroem,MMW,temp,press,rayleigh_kappa,struc_len,freq_len)

          use constants_block
          implicit none

          ! I/O
          integer, intent(in)                         :: freq_len, struc_len
          character(len=20), intent(in)                    :: spec
          double precision, intent(in)                :: lambda_angstroem(freq_len), abund(struc_len), &
               MMW(struc_len), temp(struc_len), press(struc_len)
          double precision, intent(out)               :: rayleigh_kappa(freq_len,struc_len)

          ! Internal
          integer                                     :: i_str, i_freq
          double precision                            :: lambda_cm(freq_len), &
               lamb_inv(freq_len), alpha_pol, lamb_inv_use
          double precision                            :: a0, a1, a2, a3, a4, a5, &
               a6, a7, luv, lir, l(freq_len), &
               d(struc_len), T(struc_len), retVal, retValMin, retValMax, mass_h2o, &
               nm1, fk, scale, mass_co2, &
               mass_o2, mass_n2, A, B, C, mass_co, nfr_co, &
               mass_ch4, nfr_ch4

          rayleigh_kappa = 0d0

          if (trim(adjustl(spec)) == 'H2') then

             ! H2 Rayleigh according to dalgarno & williams (1962)
             do i_str = 1, struc_len

                if (abund(i_str) > 1d-60) then
                   rayleigh_kappa(:,i_str) = rayleigh_kappa(:,i_str) + &
                        (8.14d-13/lambda_angstroem**4+1.28d-6/lambda_angstroem**6+1.61d0/lambda_angstroem**8)/2d0 &
                        /1.66053892d-24*abund(i_str)
                end if

             end do

          else if (trim(adjustl(spec)) == 'He') then

             ! He Rayleigh scattering according to Chan & Dalgarno alphas (1965)
             lambda_cm = lambda_angstroem*1d-8

             do i_str = 1, struc_len
                if (abund(i_str) > 1d-60) then
                   do i_freq = 1, freq_len

                      if (lambda_cm(i_freq) >= 0.9110d-4) then
                         alpha_pol = 1.379
                      else
                         alpha_pol = 2.265983321d0 - 3.721350022d0*lambda_cm(i_freq)/1d-4 &
                              + 3.016150391d0*(lambda_cm(i_freq)/1d-4)**2d0
                      end if

                      rayleigh_kappa(i_freq,i_str) = rayleigh_kappa(i_freq,i_str) &
                           +128d0*pi**5d0/3d0/lambda_cm(i_freq)**4d0*(alpha_pol*1.482d-25)**2d0/4d0 &
                           /1.66053892d-24*abund(i_str)
                   end do
                end if
             end do

          else if (trim(adjustl(spec)) == 'H2O') then

             ! For H2O Rayleigh scattering according to Harvey et al. (1998)
             a0 = 0.244257733
             a1 = 9.74634476d-3
             a2 = -3.73234996d-3
             a3 = 2.68678472d-4
             a4 = 1.58920570d-3
             a5 = 2.45934259d-3
             a6 = 0.900704920
             a7 = -1.66626219d-2
             luv = 0.2292020d0
             lir = 5.432937d0
             mass_h2o = 18d0*amu

             lambda_cm = lambda_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm

             l = lambda_cm/1d-4/0.589d0
             d = MMW*amu*press/kB/temp*abund
             T = temp/273.15d0

             do i_str = 1, struc_len
                if (abund(i_str) > 1d-60) then
                   do i_freq = 1, freq_len

                      retVal = (a0+a1*d(i_str)+a2*T(i_str)+a3*l(i_freq)**2d0*T(i_str)+a4/l(i_freq)**2d0 &
                           + a5/(l(i_freq)**2d0-luv**2d0) + a6/(l(i_freq)**2d0-lir**2d0) + &
                           a7*d(i_str)**2d0)*d(i_str)

                      retValMin = (a0+a1*d(i_str)+a2*T(i_str)+a3*(0.2d0/0.589d0)**2d0*T(i_str)+a4/(0.2d0/0.589d0)**2d0 &
                           + a5/((0.2d0/0.589d0)**2d0-luv**2d0) + a6/((0.2d0/0.589d0)**2d0-lir**2d0) + &
                           a7*d(i_str)**2d0)*d(i_str)

                      retValMax = (a0+a1*d(i_str)+a2*T(i_str)+a3*(1.1d0/0.589d0)**2d0*T(i_str)+a4/(1.1d0/0.589d0)**2d0 &
                           + a5/((1.1d0/0.589d0)**2d0-luv**2d0) + a6/((1.1d0/0.589d0)**2d0-lir**2d0) + &
                           a7*d(i_str)**2d0)*d(i_str)

                      if ((lambda_cm(i_freq)/1d-4 > 0.2d0) .and. (lambda_cm(i_freq)/1d-4 < 1.1d0)) then
                         nm1 = sqrt((1d0+2d0*retVal)/(1d0-retVal))
                      else if (lambda_cm(i_freq)/1d-4 >= 1.1d0) then
                         nm1 = sqrt((1.+2.*retValMax)/(1.-retValMax))
                      else
                         nm1 = sqrt((1.+2.*retValMin)/(1.-retValMin))
                      end if

                      nm1 = nm1 - 1d0
                      fk = 1.0

                      retVal = 24d0*pi**3d0*lamb_inv(i_freq)**4d0/(d(i_str)/18d0/amu)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_h2o * &
                           abund(i_str)

                      if (.NOT. ISNAN(retVal)) then
                         rayleigh_kappa(i_freq,i_str) = rayleigh_kappa(i_freq,i_str) &
                              + retVal
                      end if

                   end do
                end if
             end do

          else if (trim(adjustl(spec)) == 'CO2') then

             ! CO2 Rayleigh scattering according to Sneep & Ubachs (2004)
             d = MMW*amu*press/kB/temp*abund

             lambda_cm = lambda_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm
             mass_co2 = 44d0*amu

             do i_str = 1, struc_len
                if (abund(i_str) > 1d-60) then
                   scale = d(i_str)/44d0/amu/sneep_ubachs_n
                   do i_freq = 1, freq_len

                      nm1 = 1d-3*1.1427d6*( 5799.25d0/max(20d0**2d0,128908.9d0**2d0-lamb_inv(i_freq)**2d0) + &
                           120.05d0/max(20d0**2d0,89223.8d0**2d0-lamb_inv(i_freq)**2d0) + &
                           5.3334d0/max(20d0**2d0,75037.5d0**2d0-lamb_inv(i_freq)**2d0) + &
                           4.3244/max(20d0**2d0,67837.7d0**2d0-lamb_inv(i_freq)**2d0) + &
                           0.1218145d-4/max(20d0**2d0,2418.136d0**2d0-lamb_inv(i_freq)**2d0))
                      nm1 = nm1 * scale
                      fk = 1.1364+25.3d-12*lamb_inv(i_freq)**2d0
                      rayleigh_kappa(i_freq,i_str) = rayleigh_kappa(i_freq,i_str) &
                           + 24d0*pi**3d0*lamb_inv(i_freq)**4d0/(scale*sneep_ubachs_n)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_co2 * &
                           abund(i_str)

                   end do
                end if
             end do

          else if (trim(adjustl(spec)) == 'O2') then

             ! O2 Rayleigh scattering according to Thalman et al. (2014).
             ! Also see their erratum!
             d = MMW*amu*press/kB/temp*abund

             lambda_cm = lambda_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm
             mass_o2 = 32d0*amu

             do i_str = 1, struc_len
                if (abund(i_str) > 1d-60) then
                   scale = d(i_str)/mass_o2/2.68678d19

                   do i_freq = 1, freq_len

                      if (lamb_inv(i_freq) > 18315d0) then
                         A = 20564.8d0
                         B = 2.480899d13
                      else
                         A = 21351.1d0
                         B = 2.18567d13
                      end if
                      C = 4.09d9

                      nm1 = 1d-8*(A+B/(C-lamb_inv(i_freq)**2d0))
                      nm1 = nm1 !* scale
                      fk = 1.096d0+1.385d-11*lamb_inv(i_freq)**2d0+1.448d-20*lamb_inv(i_freq)**4d0
                      rayleigh_kappa(i_freq,i_str) = rayleigh_kappa(i_freq,i_str) &
                           + 24d0*pi**3d0*lamb_inv(i_freq)**4d0/(2.68678d19)**2d0* & !(d(i_str)/mass_o2)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_o2 * &
                           abund(i_str)

                   end do
                end if
             end do

          else if (trim(adjustl(spec)) == 'N2') then

             ! N2 Rayleigh scattering according to Thalman et al. (2014).
             ! Also see their erratum!
             d = MMW*amu*press/kB/temp*abund

             lambda_cm = lambda_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm
             mass_n2 = 34d0*amu

             do i_str = 1, struc_len
                if (abund(i_str) > 1d-60) then
                   scale = d(i_str)/mass_n2/2.546899d19

                   do i_freq = 1, freq_len

                      if (lamb_inv(i_freq) > 4860d0) then

                         if (lamb_inv(i_freq) > 21360d0) then
                            A = 5677.465d0
                            B = 318.81874d12
                            C = 14.4d9
                         else
                            A = 6498.2d0
                            B = 307.43305d12
                            C = 14.4d9
                         end if

                         nm1 = 1d-8*(A+B/(C-lamb_inv(i_freq)**2d0))
                         nm1 = nm1 !* scale
                         fk = 1.034d0+3.17d-12*lamb_inv(i_freq)**2d0
                         rayleigh_kappa(i_freq,i_str) = rayleigh_kappa(i_freq,i_str) &
                              + 24d0*pi**3d0*lamb_inv(i_freq)**4d0/(2.546899d19)**2d0* & !(d(i_str)/mass_n2)**2d0* &
                              (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_n2 * &
                              abund(i_str)

                      end if

                   end do
                end if
             end do

          else if (trim(adjustl(spec)) == 'CO') then

             ! CO Rayleigh scattering according to Sneep & Ubachs (2004)

             lambda_cm = lambda_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm

             d = MMW*amu*press/kB/temp*abund

             do i_str = 1, struc_len

                if (abund(i_str) > 1d-60) then

                   scale = d(i_str)/28d0/amu/sneep_ubachs_n
                   nfr_co = d(i_str)/28d0/amu
                   mass_co = 28d0*amu

                   do i_freq = 1, freq_len

                      lamb_inv_use = lamb_inv(i_freq)
                      if (lambda_cm(i_freq)/1e-4 < 0.168d0) then
                         lamb_inv_use = 1d0/0.168d-4
                      end if
                      nm1 = (22851d0 + 0.456d12/(71427d0**2d0-lamb_inv_use**2d0))*1d-8
                      nm1 = nm1 * scale
                      fk = 1.016d0

                      rayleigh_kappa(i_freq,i_str) = rayleigh_kappa(i_freq,i_str) &
                           + 24d0*pi**3d0*lamb_inv(i_freq)**4d0/(nfr_co)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_co * &
                           abund(i_str)

                   end do
                end if

             end do

          else if (trim(adjustl(spec)) == 'CH4') then

             ! CH4 Rayleigh scattering according to Sneep & Ubachs (2004)

             lambda_cm = lambda_angstroem*1d-8
             lamb_inv = 1d0/lambda_cm

             d = MMW*amu*press/kB/temp*abund

             do i_str = 1, struc_len

                if (abund(i_str) > 1d-60) then

                   scale = d(i_str)/16d0/amu/sneep_ubachs_n
                   nfr_ch4 = d(i_str)/16d0/amu
                   mass_ch4 = 16d0*amu

                   do i_freq = 1, freq_len

                      nm1 = (46662d0 + 4.02d-6*lamb_inv(i_freq)**2d0)*1d-8
                      nm1 = nm1 * scale
                      fk = 1.0
                      rayleigh_kappa(i_freq,i_str) = rayleigh_kappa(i_freq,i_str) &
                           + 24d0*pi**3d0*lamb_inv(i_freq)**4d0/(nfr_ch4)**2d0* &
                           (((nm1+1d0)**2d0-1d0)/((nm1+1d0)**2d0+2d0))**2d0*fk / mass_ch4 * &
                           abund(i_str)

                   end do
                end if

             end do


          end if

        end subroutine add_rayleigh
        

        subroutine calc_transm_spec_contr(total_kappa,temp,press,gravity,mmw,P0_bar,R_pl, &
             w_gauss,transm_in,scat,continuum_opa_scat,var_grav,contr_tr,freq_len,struc_len,g_len,N_species)
            ! """
            ! Subroutine to calculate the contribution function of the transmission spectrum
            ! """
            use constants_block
            
            implicit none
            
            logical, intent(in)                         :: scat
            logical, intent(in)                         :: var_grav
            integer, intent(in)                         :: freq_len, struc_len, g_len, N_species
            double precision, intent(in)                :: P0_bar, R_pl
            double precision, intent(in)                :: temp(struc_len), press(struc_len), mmw(struc_len)
            double precision, intent(in)                :: total_kappa(g_len,freq_len,N_species,struc_len)
            double precision, intent(in)                :: gravity
            double precision, intent(in)                :: w_gauss(g_len), continuum_opa_scat(freq_len,struc_len)
            double precision, intent(in)                :: transm_in(freq_len)
            double precision, intent(out)               :: contr_tr(struc_len,freq_len)
            
            integer                                     :: i_str, i_freq,  i_spec, j_str, i_leave_str
            double precision                            :: P0_cgs, rho(struc_len), radius(struc_len)
            double precision                            :: alpha_t2(g_len,freq_len,N_species,struc_len-1)
            double precision                            :: t_graze(g_len,freq_len,N_species,struc_len), s_1, s_2, &
               t_graze_wlen_int(struc_len,freq_len), alpha_t2_scat(freq_len,struc_len-1), &
               t_graze_scat(freq_len,struc_len), total_kappa_use(g_len,freq_len,N_species,struc_len), &
               continuum_opa_scat_use(freq_len,struc_len), transm(freq_len)
            
            ! Convert reference pressure to cgs
            P0_cgs = P0_bar * 1d6
            
            ! Calculate density
            rho = mmw * amu * press / kB / temp
            
            ! Calculate planetary radius (in cm), assuming hydrostatic equilibrium
            call calc_radius(struc_len, press, gravity, rho, P0_cgs, R_pl, var_grav, radius)
            
            do i_leave_str = 1, struc_len
                transm = 0d0
                t_graze = 0d0
                t_graze_scat = 0d0
                
                continuum_opa_scat_use = continuum_opa_scat
                total_kappa_use = total_kappa
                total_kappa_use(:, :, :, i_leave_str) = 0d0
                continuum_opa_scat_use(:, i_leave_str) = 0d0
                
                ! Calc. mean free paths across grazing distances
                do i_str = 1, struc_len-1
                    alpha_t2(:, :, :, i_str) = &
                        total_kappa_use(:, :, :, i_str) * rho(i_str) &
                        + total_kappa_use(:, :, :, i_str + 1) * rho(i_str + 1)
                end do
                
                if (scat) then
                    do i_str = 1, struc_len - 1
                        alpha_t2_scat(:,i_str) = continuum_opa_scat_use(:, i_str) * rho(i_str) &
                            + continuum_opa_scat_use(:, i_str + 1) * rho(i_str + 1)
                    end do
                end if
                
                ! Cacuclate grazing rays optical depths
                do i_str = 2, struc_len
                    s_1 = sqrt(radius(1) ** 2d0 - radius(i_str) ** 2d0)
                    
                    do j_str = 1, i_str-1
                        if (j_str > 1) then
                            s_1 = s_2
                        end if
                       
                        s_2 = sqrt(radius(j_str + 1) ** 2d0 - radius(i_str) ** 2d0)
                        t_graze(:, :, :, i_str) = t_graze(:, :, :, i_str) + alpha_t2(:, :, :, j_str) * (s_1 - s_2)
                    end do
                end do
                
                if (scat) then
                    do i_str = 2, struc_len
                        s_1 = sqrt(radius(1) ** 2d0 - radius(i_str) ** 2d0)
                       
                        do j_str = 1, i_str - 1
                            if (j_str > 1) then
                                s_1 = s_2
                            end if
                        
                            s_2 = sqrt(radius(j_str + 1) ** 2d0 - radius(i_str) ** 2d0)
                            t_graze_scat(:, i_str) = t_graze_scat(:, i_str) + alpha_t2_scat(:, j_str) * (s_1 - s_2)
                        end do
                    end do
                end if
                
                ! Calculate transmissions, update tau array to store these
                t_graze = exp(-t_graze)
                
                if (scat) then
                    t_graze_scat = exp(-t_graze_scat)
                end if
                
                t_graze_wlen_int = 1d0
                
                ! Wlen (in g-space) integrate transmissions
                do i_str = 2, struc_len ! i_str=1 t_grazes are 1 anyways
                    do i_spec = 1, N_species
                        do i_freq = 1, freq_len
                            t_graze_wlen_int(i_str,i_freq) = t_graze_wlen_int(i_str,i_freq)&
                                * sum(t_graze(:, i_freq, i_spec, i_str) * w_gauss)
                          
                            if (scat .and. (i_spec == 1)) then
                                t_graze_wlen_int(i_str,i_freq) = t_graze_wlen_int(i_str, i_freq) &
                                    * t_graze_scat(i_freq, i_str)
                            end if
                        end do
                    end do
                end do
                
                ! Get effective area fraction from transmission
                t_graze_wlen_int = 1d0 - t_graze_wlen_int
                
                ! Caculate planets effectice area (leaving out pi, because we want the radius in the end)
                do i_freq = 1, freq_len
                    do i_str = 2, struc_len
                        transm(i_freq) = &
                            transm(i_freq) &
                            + (&
                                t_graze_wlen_int(i_str - 1, i_freq) * radius(i_str-1) &
                                + t_graze_wlen_int(i_str, i_freq) * radius(i_str) &
                            ) &
                            * (radius(i_str - 1) - radius(i_str))
                    end do
                end do

                ! Get radius
                transm = transm+radius(struc_len) ** 2d0
                contr_tr(i_leave_str, :) = transm_in - transm
            end do
            
            do i_freq = 1, freq_len
                contr_tr(:, i_freq) = contr_tr(:, i_freq) / sum(contr_tr(:, i_freq))
            end do
        end subroutine calc_transm_spec_contr


        function hansen_size_dndr(r,a,b,k)

           implicit none
           ! I/O
           double precision :: r, a, b, k
           double precision :: hansen_size_dndr

           hansen_size_dndr = (((1d0-(3d0*b))*k*r**((1d0-(3d0*b))/(b-1d0)) * &
                          exp(-1d0*r/(a*b)))/b) -((k*r**((1d0-(3d0*b))/(b))*exp(-1d0*r/(a*b)))/(a*b))
        end function hansen_size_dndr

        !!$ Subroutine to calculate cloud opacities
        subroutine calc_cloud_opas(rho,rho_p,cloud_mass_fracs,r_g,sigma_n,cloud_rad_bins,cloud_radii, &
           cloud_specs_abs_opa,cloud_specs_scat_opa,cloud_aniso, &
           cloud_abs_opa_TOT,cloud_scat_opa_TOT,cloud_red_fac_aniso_TOT, &
           struc_len,N_cloud_spec,N_cloud_rad_bins, N_cloud_lambda_bins)

          use constants_block
          implicit none

          ! I/O
          integer, intent(in) :: struc_len, N_cloud_spec, N_cloud_rad_bins, N_cloud_lambda_bins
          double precision, intent(in) :: rho(struc_len), rho_p(N_cloud_spec)
          double precision, intent(in) :: cloud_mass_fracs(struc_len,N_cloud_spec),r_g(struc_len,N_cloud_spec)
          double precision, intent(in) :: sigma_n
          double precision, intent(in) :: cloud_rad_bins(N_cloud_rad_bins+1), cloud_radii(N_cloud_rad_bins)
          double precision, intent(in) :: cloud_specs_abs_opa(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec), &
               cloud_specs_scat_opa(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec), &
               cloud_aniso(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec)

          double precision, intent(out) :: cloud_abs_opa_TOT(N_cloud_lambda_bins,struc_len), &
               cloud_scat_opa_TOT(N_cloud_lambda_bins,struc_len), &
               cloud_red_fac_aniso_TOT(N_cloud_lambda_bins,struc_len)


          ! internal
          integer :: i_struc, i_spec, i_lamb
          double precision :: N, dndr(N_cloud_rad_bins), integrand_abs(N_cloud_rad_bins), &
               integrand_scat(N_cloud_rad_bins), add_abs, add_scat, integrand_aniso(N_cloud_rad_bins), add_aniso

          !~~~~~~~~~~~~~~~~
          cloud_abs_opa_TOT = 0d0
          cloud_scat_opa_TOT = 0d0
          cloud_red_fac_aniso_TOT = 0d0

          do i_struc = 1, struc_len
             do i_spec = 1, N_cloud_spec
                   do i_lamb = 1, N_cloud_lambda_bins
                      N = 3d0*cloud_mass_fracs(i_struc,i_spec)*rho(i_struc)/4d0/pi/rho_p(i_spec)/ &
                          r_g(i_struc,i_spec)**3d0*exp(-9d0/2d0*log(sigma_n)**2d0)

                      dndr = N/(cloud_radii*sqrt(2d0*pi)*log(sigma_n))* &
                          exp(-log(cloud_radii/r_g(i_struc,i_spec))**2d0/(2d0*log(sigma_n)**2d0))


                      integrand_abs = 4d0*pi/3d0*cloud_radii**3d0*rho_p(i_spec)*dndr* &
                           cloud_specs_abs_opa(:,i_lamb,i_spec)
                      integrand_scat = 4d0*pi/3d0*cloud_radii**3d0*rho_p(i_spec)*dndr* &
                           cloud_specs_scat_opa(:,i_lamb,i_spec)
                      integrand_aniso = integrand_scat*(1d0-cloud_aniso(:,i_lamb,i_spec))

                      add_abs = sum(integrand_abs*(cloud_rad_bins(2:N_cloud_rad_bins+1)- &
                           cloud_rad_bins(1:N_cloud_rad_bins)))
                      cloud_abs_opa_TOT(i_lamb,i_struc) = cloud_abs_opa_TOT(i_lamb,i_struc) + &
                           add_abs

                      add_scat = sum(integrand_scat*(cloud_rad_bins(2:N_cloud_rad_bins+1)- &
                           cloud_rad_bins(1:N_cloud_rad_bins)))
                      cloud_scat_opa_TOT(i_lamb,i_struc) = cloud_scat_opa_TOT(i_lamb,i_struc) + &
                           add_scat

                      add_aniso = sum(integrand_aniso*(cloud_rad_bins(2:N_cloud_rad_bins+1)- &
                           cloud_rad_bins(1:N_cloud_rad_bins)))
                      cloud_red_fac_aniso_TOT(i_lamb,i_struc) = cloud_red_fac_aniso_TOT(i_lamb,i_struc) + &
                           add_aniso

                   end do

             end do

             do i_lamb = 1, N_cloud_lambda_bins
                if (cloud_scat_opa_TOT(i_lamb,i_struc) > 1d-200) then
                   cloud_red_fac_aniso_TOT(i_lamb,i_struc) = cloud_red_fac_aniso_TOT(i_lamb,i_struc)/ &
                             cloud_scat_opa_TOT(i_lamb,i_struc)
                else
                   cloud_red_fac_aniso_TOT(i_lamb,i_struc) = 0d0
                end if
             end do

             cloud_abs_opa_TOT(:,i_struc) = cloud_abs_opa_TOT(:,i_struc)/rho(i_struc)
             cloud_scat_opa_TOT(:,i_struc) = cloud_scat_opa_TOT(:,i_struc)/rho(i_struc)

          end do

        end subroutine calc_cloud_opas


        subroutine calc_hansen_opas(rho,rho_p,cloud_mass_fracs,a_h,b_h,cloud_rad_bins, &
           cloud_radii,cloud_specs_abs_opa,cloud_specs_scat_opa,cloud_aniso, &
           cloud_abs_opa_TOT,cloud_scat_opa_TOT,cloud_red_fac_aniso_TOT, &
           struc_len,N_cloud_spec,N_cloud_rad_bins, N_cloud_lambda_bins)
            ! """
            ! Subroutine to calculate cloud opacities.
            ! """
            use constants_block
            implicit none

            integer, intent(in) :: struc_len, N_cloud_spec, N_cloud_rad_bins, N_cloud_lambda_bins
            double precision, intent(in) :: rho(struc_len), rho_p(N_cloud_spec)
            double precision, intent(in) :: cloud_mass_fracs(struc_len,N_cloud_spec), &
                a_h(struc_len,N_cloud_spec), b_h(struc_len,N_cloud_spec)
            double precision, intent(in) :: cloud_rad_bins(N_cloud_rad_bins+1), cloud_radii(N_cloud_rad_bins)
            double precision, intent(in) :: cloud_specs_abs_opa(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec), &
                cloud_specs_scat_opa(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec), &
                cloud_aniso(N_cloud_rad_bins,N_cloud_lambda_bins,N_cloud_spec)
            double precision, intent(out) :: cloud_abs_opa_TOT(N_cloud_lambda_bins,struc_len), &
                cloud_scat_opa_TOT(N_cloud_lambda_bins,struc_len), &
                cloud_red_fac_aniso_TOT(N_cloud_lambda_bins,struc_len)

            integer :: i_struc, i_spec, i_lamb, i_cloud
            double precision :: N, dndr(N_cloud_rad_bins), integrand_abs(N_cloud_rad_bins), mass_to_vol, &
                integrand_scat(N_cloud_rad_bins), add_abs, add_scat, integrand_aniso(N_cloud_rad_bins), add_aniso, &
                dndr_scale

            cloud_abs_opa_TOT = 0d0
            cloud_scat_opa_TOT = 0d0
            cloud_red_fac_aniso_TOT = 0d0

            do i_struc = 1, struc_len
                do i_spec = 1, N_cloud_spec
                    do i_lamb = 1, N_cloud_lambda_bins
                        mass_to_vol = 0.75d0 * cloud_mass_fracs(i_struc, i_spec) * rho(i_struc) / pi / rho_p(i_spec)

                        N = mass_to_vol / (&
                            a_h(i_struc, i_spec) ** 3d0 * (b_h(i_struc,i_spec) -1d0) &
                            * (2d0 * b_h(i_struc,i_spec) - 1d0) &
                        )
                        dndr_scale = &
                            log(N) + log(a_h(i_struc, i_spec) * b_h(i_struc, i_spec)) &
                                * ((2d0 * (b_h(i_struc, i_spec)) - 1d0) / b_h(i_struc, i_spec)) &
                                - log(gamma((1d0 - 2d0 * b_h(i_struc, i_spec)) / b_h(i_struc, i_spec)))

                        do i_cloud = 1, N_cloud_rad_bins
                            dndr(i_cloud) = dndr_scale + hansen_size_nr(&
                                cloud_radii(i_cloud), &
                                a_h(i_struc,i_spec), &
                                b_h(i_struc,i_spec) &
                            )
                        end do

                        dndr = exp(dndr)

                        integrand_abs = 0.75d0 * pi * cloud_radii ** 3d0 * rho_p(i_spec) * dndr &
                            * cloud_specs_abs_opa(:,i_lamb,i_spec)
                        integrand_scat = 0.75d0 * pi * cloud_radii ** 3d0 * rho_p(i_spec) * dndr &
                            * cloud_specs_scat_opa(:,i_lamb,i_spec)
                        integrand_aniso = integrand_scat * (1d0 - cloud_aniso(:, i_lamb, i_spec))

                        add_abs = sum(&
                            integrand_abs &
                            * (cloud_rad_bins(2:N_cloud_rad_bins + 1) - cloud_rad_bins(1:N_cloud_rad_bins)) &
                        )
                        cloud_abs_opa_TOT(i_lamb,i_struc) = cloud_abs_opa_TOT(i_lamb, i_struc) + add_abs

                        add_scat = sum(&
                            integrand_scat &
                            * (cloud_rad_bins(2:N_cloud_rad_bins + 1) - cloud_rad_bins(1:N_cloud_rad_bins)) &
                        )
                        cloud_scat_opa_TOT(i_lamb, i_struc) = cloud_scat_opa_TOT(i_lamb, i_struc) + add_scat

                        add_aniso = sum(&
                            integrand_aniso &
                            * (cloud_rad_bins(2:N_cloud_rad_bins+1) - cloud_rad_bins(1:N_cloud_rad_bins)) &
                        )
                        cloud_red_fac_aniso_TOT(i_lamb, i_struc) = &
                            cloud_red_fac_aniso_TOT(i_lamb, i_struc) + add_aniso
                    end do
                end do

                do i_lamb = 1, N_cloud_lambda_bins
                    if (cloud_scat_opa_TOT(i_lamb,i_struc) > 1d-200) then
                        cloud_red_fac_aniso_TOT(i_lamb, i_struc) = &
                            cloud_red_fac_aniso_TOT(i_lamb, i_struc) / cloud_scat_opa_TOT(i_lamb, i_struc)
                    else
                        cloud_red_fac_aniso_TOT(i_lamb, i_struc) = 0d0
                    end if
                end do

                cloud_abs_opa_TOT(:, i_struc) = cloud_abs_opa_TOT(:, i_struc) / rho(i_struc)
                cloud_scat_opa_TOT(:, i_struc) = cloud_scat_opa_TOT(:, i_struc) / rho(i_struc)
            end do

            contains
                function hansen_size_nr(r,a,b)
                    implicit none

                    double precision :: r, a, b
                    double precision :: hansen_size_nr

                    hansen_size_nr = log(r) * (1d0 - 3d0 * b) / b - 1d0 * r / (a * b)
                end function hansen_size_nr
        end subroutine calc_hansen_opas
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        !!$ Interpolate cloud opacities to actual radiative transfer wavelength grid

        subroutine interp_integ_cloud_opas(cloud_abs_opa_TOT,cloud_scat_opa_TOT, &
             cloud_red_fac_aniso_TOT,cloud_lambdas,HIT_border_freqs,HIT_kappa_tot_g_approx, &
             HIT_kappa_tot_g_approx_scat,red_fac_aniso_final, HIT_kappa_tot_g_approx_scat_unred, &
             N_cloud_lambda_bins,struc_len,HIT_coarse_borders)

          use constants_block
          implicit none
          ! I/O
          integer, intent(in)           :: N_cloud_lambda_bins,struc_len,HIT_coarse_borders
          double precision, intent(in)  :: cloud_abs_opa_TOT(N_cloud_lambda_bins,struc_len), &
               cloud_scat_opa_TOT(N_cloud_lambda_bins,struc_len), &
               cloud_red_fac_aniso_TOT(N_cloud_lambda_bins,struc_len), cloud_lambdas(N_cloud_lambda_bins), &
               HIT_border_freqs(HIT_coarse_borders)
          double precision, intent(out) :: HIT_kappa_tot_g_approx(HIT_coarse_borders-1,struc_len), &
               HIT_kappa_tot_g_approx_scat(HIT_coarse_borders-1,struc_len), &
               red_fac_aniso_final(HIT_coarse_borders-1,struc_len), &
               HIT_kappa_tot_g_approx_scat_unred(HIT_coarse_borders-1,struc_len)

          ! internal
          double precision :: kappa_integ(struc_len), kappa_scat_integ(struc_len), red_fac_aniso_integ(struc_len), &
               kappa_tot_integ(HIT_coarse_borders-1,struc_len), kappa_tot_scat_integ(HIT_coarse_borders-1,struc_len)
          integer          :: HIT_i_lamb
          double precision :: HIT_border_lamb(HIT_coarse_borders)
          integer          :: intp_index_small_min, intp_index_small_max, &
               new_small_ind

          HIT_kappa_tot_g_approx = 0d0
          HIT_kappa_tot_g_approx_scat = 0d0
          HIT_kappa_tot_g_approx_scat_unred = 0d0


          HIT_border_lamb = c_l/HIT_border_freqs
          red_fac_aniso_final = 0d0

          kappa_tot_integ = 0d0
          kappa_tot_scat_integ = 0d0

          do HIT_i_lamb = 1, HIT_coarse_borders-1

             intp_index_small_min = MIN(MAX(INT((log10(HIT_border_lamb(HIT_i_lamb))-log10(cloud_lambdas(1))) / &
                  log10(cloud_lambdas(N_cloud_lambda_bins)/cloud_lambdas(1))*dble(N_cloud_lambda_bins-1) &
                  +1d0),1),N_cloud_lambda_bins-1)

             intp_index_small_max = MIN(MAX(INT((log10(HIT_border_lamb(HIT_i_lamb+1))-log10(cloud_lambdas(1))) / &
                  log10(cloud_lambdas(N_cloud_lambda_bins)/cloud_lambdas(1))*dble(N_cloud_lambda_bins-1) &
                  +1d0),1),N_cloud_lambda_bins-1)

             kappa_integ = 0d0
             kappa_scat_integ = 0d0
             red_fac_aniso_integ = 0d0

             if ((intp_index_small_max-intp_index_small_min) == 0) then

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_abs_opa_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),HIT_border_lamb(HIT_i_lamb+1),kappa_integ)

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_scat_opa_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),HIT_border_lamb(HIT_i_lamb+1),kappa_scat_integ)

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_red_fac_aniso_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),HIT_border_lamb(HIT_i_lamb+1),red_fac_aniso_integ)

             else if ((intp_index_small_max-intp_index_small_min) == 1) then

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_abs_opa_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),cloud_lambdas(intp_index_small_min+1),kappa_integ)

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_scat_opa_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),cloud_lambdas(intp_index_small_min+1),kappa_scat_integ)

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_red_fac_aniso_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),cloud_lambdas(intp_index_small_min+1),&
                     red_fac_aniso_integ)

                call integ_kaps(intp_index_small_max,N_cloud_lambda_bins,struc_len,cloud_abs_opa_TOT, &
                     cloud_lambdas,cloud_lambdas(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),kappa_integ)

                call integ_kaps(intp_index_small_max,N_cloud_lambda_bins,struc_len,cloud_scat_opa_TOT, &
                     cloud_lambdas,cloud_lambdas(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),kappa_scat_integ)

                call integ_kaps(intp_index_small_max,N_cloud_lambda_bins,struc_len,cloud_red_fac_aniso_TOT, &
                     cloud_lambdas,cloud_lambdas(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),&
                     red_fac_aniso_integ)

             else

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_abs_opa_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),cloud_lambdas(intp_index_small_min+1),kappa_integ)

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_scat_opa_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),cloud_lambdas(intp_index_small_min+1),kappa_scat_integ)

                call integ_kaps(intp_index_small_min,N_cloud_lambda_bins,struc_len,cloud_red_fac_aniso_TOT, &
                     cloud_lambdas,HIT_border_lamb(HIT_i_lamb),cloud_lambdas(intp_index_small_min+1),&
                     red_fac_aniso_integ)

                new_small_ind = intp_index_small_min+1
                do while (intp_index_small_max-new_small_ind /= 0)

                   call integ_kaps(new_small_ind,N_cloud_lambda_bins,struc_len,cloud_abs_opa_TOT, &
                        cloud_lambdas,cloud_lambdas(new_small_ind),cloud_lambdas(new_small_ind+1),kappa_integ)

                   call integ_kaps(new_small_ind,N_cloud_lambda_bins,struc_len,cloud_scat_opa_TOT, &
                        cloud_lambdas,cloud_lambdas(new_small_ind),cloud_lambdas(new_small_ind+1),kappa_scat_integ)

                   call integ_kaps(new_small_ind,N_cloud_lambda_bins,struc_len,cloud_red_fac_aniso_TOT, &
                        cloud_lambdas,cloud_lambdas(new_small_ind),cloud_lambdas(new_small_ind+1),red_fac_aniso_integ)

                   new_small_ind = new_small_ind+1

                end do

                call integ_kaps(intp_index_small_max,N_cloud_lambda_bins,struc_len,cloud_abs_opa_TOT, &
                     cloud_lambdas,cloud_lambdas(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),kappa_integ)

                call integ_kaps(intp_index_small_max,N_cloud_lambda_bins,struc_len,cloud_scat_opa_TOT, &
                     cloud_lambdas,cloud_lambdas(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),kappa_scat_integ)

                call integ_kaps(intp_index_small_max,N_cloud_lambda_bins,struc_len,cloud_red_fac_aniso_TOT, &
                     cloud_lambdas,cloud_lambdas(intp_index_small_max),HIT_border_lamb(HIT_i_lamb+1),&
                     red_fac_aniso_integ)

             end if

             kappa_integ = kappa_integ/(HIT_border_lamb(HIT_i_lamb+1)-HIT_border_lamb(HIT_i_lamb))
             kappa_scat_integ = kappa_scat_integ/(HIT_border_lamb(HIT_i_lamb+1)-HIT_border_lamb(HIT_i_lamb))
             red_fac_aniso_integ = red_fac_aniso_integ/(HIT_border_lamb(HIT_i_lamb+1)-HIT_border_lamb(HIT_i_lamb))

             kappa_tot_integ(HIT_i_lamb,:) = kappa_integ
             kappa_tot_scat_integ(HIT_i_lamb,:) = kappa_scat_integ

             HIT_kappa_tot_g_approx(HIT_i_lamb,:) = HIT_kappa_tot_g_approx(HIT_i_lamb,:) + &
                  kappa_integ
             HIT_kappa_tot_g_approx_scat(HIT_i_lamb,:) = HIT_kappa_tot_g_approx_scat(HIT_i_lamb,:) + &
                  kappa_integ + kappa_scat_integ*red_fac_aniso_integ
             HIT_kappa_tot_g_approx_scat_unred(HIT_i_lamb,:) = HIT_kappa_tot_g_approx_scat_unred(HIT_i_lamb,:) + &
                  kappa_integ + kappa_scat_integ

             red_fac_aniso_final(HIT_i_lamb,:) = red_fac_aniso_final(HIT_i_lamb,:) + red_fac_aniso_integ

          end do

        end subroutine interp_integ_cloud_opas

        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################
        !!$ #########################################################################

        subroutine integ_kaps(intp_ind,N_cloud_lambda_bins,struc_len,kappa,lambda,l_bord1,l_bord2,kappa_integ)
          implicit none
          integer, intent(in) :: intp_ind,N_cloud_lambda_bins,struc_len
          double precision, intent(in) :: lambda(N_cloud_lambda_bins), kappa(N_cloud_lambda_bins,struc_len)
          double precision, intent(in) :: l_bord1,l_bord2
          double precision, intent(out) :: kappa_integ(struc_len)

          ! This subroutine calculates the integral of a linearly interpolated function kappa.

          kappa_integ = kappa_integ + kappa(intp_ind,:)*(l_bord2-l_bord1) + (kappa(intp_ind+1,:)-kappa(intp_ind,:))/ &
               (lambda(intp_ind+1)-lambda(intp_ind))* &
               0.5d0*((l_bord2-lambda(intp_ind))**2d0-(l_bord1-lambda(intp_ind))**2d0)

        end subroutine integ_kaps

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine get_rg_N(gravity,rho,rho_p,temp,MMW,frain, &
             sigma_n,Kzz,r_g,struc_len,N_cloud_spec)

          use constants_block
          implicit none
          ! I/O
          integer, intent(in)  :: struc_len, N_cloud_spec
          double precision, intent(in) :: gravity, rho(struc_len), rho_p(N_cloud_spec), temp(struc_len), &
               MMW(struc_len), frain(N_cloud_spec), &
               sigma_n, Kzz(struc_len)
          double precision, intent(out) :: r_g(struc_len,N_cloud_spec)
          ! Internal
          integer, parameter :: N_fit = 100
          integer          :: i_str, i_spec, i_rad
          double precision :: w_star(struc_len), H(struc_len)
          double precision :: r_w(struc_len,N_cloud_spec), alpha(struc_len,N_cloud_spec)
          double precision :: rad(N_fit), vel(N_fit), f_fill(N_cloud_spec)
          double precision :: a, b

          H = kB*temp/(MMW*amu*gravity)
          w_star = Kzz/H

          f_fill = 1d0

          do i_str = 1, struc_len
             do i_spec = 1, N_cloud_spec
                r_w(i_str,i_spec) = bisect_particle_rad(1d-16,1d2,gravity,rho(i_str), &
                     rho_p(i_spec),temp(i_str),MMW(i_str),w_star(i_str))
                if (r_w(i_str,i_spec) > 1d-16) then
                   if (frain(i_spec) > 1d0) then
                      do i_rad = 1, N_fit
                         rad(i_rad) = r_w(i_str,i_spec)/max(sigma_n,1.0001d0) + &
                              (r_w(i_str,i_spec)-r_w(i_str,i_spec)/max(sigma_n,1.0001d0))* &
                              dble(i_rad-1)/dble(N_fit-1)
                         call turbulent_settling_speed(rad(i_rad),gravity,rho(i_str),rho_p(i_spec),temp(i_str), &
                              MMW(i_str),vel(i_rad))
                      end do
                   else
                      do i_rad = 1, N_fit
                         rad(i_rad) = r_w(i_str,i_spec) + (r_w(i_str,i_spec)*max(sigma_n,1.0001d0)- &
                              r_w(i_str,i_spec))* &
                              dble(i_rad-1)/dble(N_fit-1)
                         call turbulent_settling_speed(rad(i_rad),gravity,rho(i_str),rho_p(i_spec),temp(i_str), &
                              MMW(i_str),vel(i_rad))
                      end do
                   end if

                   call fit_linear(log(rad), log(vel/w_star(i_str)), N_fit, a, b)

                   alpha(i_str,i_spec) = b
                   r_w(i_str,i_spec) = exp(-a/b)
                   r_g(i_str,i_spec) = r_w(i_str,i_spec) * frain(i_spec)**(1d0/alpha(i_str,i_spec))* &
                        exp(-(alpha(i_str,i_spec)+6d0)/2d0*log(sigma_n)**2d0)
                else
                   r_g(i_str,i_spec) = 1d-17
                   alpha(i_str,i_spec) = 1d0
                end if
             end do

          end do

        end subroutine get_rg_N

        subroutine get_rg_n_hansen(gravity,rho,rho_p,temp,MMW,frain, &
           b_h,Kzz,a_h,struc_len,N_cloud_spec)
            use constants_block
            
            implicit none

            integer, intent(in)  :: struc_len, N_cloud_spec
            double precision, intent(in) :: gravity, rho(struc_len), rho_p(N_cloud_spec), temp(struc_len), &
                 MMW(struc_len), frain(N_cloud_spec), &
                 b_h(struc_len,N_cloud_spec), Kzz(struc_len)
            double precision, intent(out) :: a_h(struc_len,N_cloud_spec)
            
            integer, parameter :: N_fit = 100
            doubleprecision, parameter :: x_gamma_max = 170d0  ! gamma(x_gamma_max) >~ huge(0d0)

            integer          :: i_str, i_spec, i_rad
            double precision :: w_star(struc_len), H(struc_len)
            double precision :: r_w(struc_len,N_cloud_spec), alpha(struc_len, N_cloud_spec), &
                x_gamma(struc_len, N_cloud_spec)
            double precision :: rad(N_fit), vel(N_fit)
            double precision :: a, b
    
            H = kB * temp / (MMW * amu * gravity)
            w_star = Kzz / H
            x_gamma = 1d0 + 1d0 / b_h  ! argument of the gamma function
    
            do i_str = 1, struc_len
                do i_spec = 1, N_cloud_spec
                    r_w(i_str,i_spec) = bisect_particle_rad(&
                        1d-16,&
                        1d2, &
                        gravity, &
                        rho(i_str), &
                        rho_p(i_spec), &
                        temp(i_str), &
                        MMW(i_str), &
                        w_star(i_str) &
                    )

                    if (r_w(i_str,i_spec) > 1d-16) then
                        if (frain(i_spec) > 1d0) then
                            do i_rad = 1, N_fit
                                rad(i_rad) = &
                                    r_w(i_str, i_spec) * b_h(i_str, i_spec) &
                                    + (r_w(i_str, i_spec) - r_w(i_str, i_spec) * b_h(i_str, i_spec)) &
                                    * dble(i_rad - 1) / dble(N_fit - 1)
                                
                                call turbulent_settling_speed(&
                                    rad(i_rad), &
                                    gravity, &
                                    rho(i_str), &
                                    rho_p(i_spec), &
                                    temp(i_str), &
                                    MMW(i_str), &
                                    vel(i_rad) &
                                )
                            end do
                        else
                            do i_rad = 1, N_fit
                                rad(i_rad) = &
                                    r_w(i_str,i_spec) &
                                    + (r_w(i_str,i_spec) / b_h(i_str,i_spec) - r_w(i_str,i_spec)) &
                                    * dble(i_rad - 1) / dble(N_fit - 1)
                                
                                call turbulent_settling_speed(&
                                    rad(i_rad), &
                                    gravity, &
                                    rho(i_str), &
                                    rho_p(i_spec), &
                                    temp(i_str), &
                                    MMW(i_str), &
                                    vel(i_rad) &
                                )
                            end do
                        end if
                        
                        call fit_linear(log(rad), log(vel / w_star(i_str)), N_fit, a, b)
                        
                        alpha(i_str, i_spec) = b
                        r_w(i_str,i_spec) = exp(-a / b)

                        if (x_gamma(i_str, i_spec) + alpha(i_str, i_spec) < x_gamma_max) then
                            a_h(i_str, i_spec) = &
                                (&
                                    b_h(i_str, i_spec) ** (-1d0 * alpha(i_str, i_spec)) &
                                    * r_w(i_str, i_spec) ** alpha(i_str, i_spec) * frain(i_spec) &
                                    * (&
                                        b_h(i_str, i_spec) ** 3d0 &
                                        * b_h(i_str, i_spec) ** alpha(i_str, i_spec) &
                                        - b_h(i_str,i_spec) + 1d0 &
                                    )&
                                    * gamma(x_gamma(i_str, i_spec)) &
                                    / (&
                                        (&
                                            b_h(i_str,i_spec) * alpha(i_str,i_spec) + 2d0 * b_h(i_str, i_spec) + 1d0 &
                                        )&
                                        * gamma(x_gamma(i_str, i_spec) + alpha(i_str, i_spec))&
                                    )&
                                ) ** (1d0 / alpha(i_str, i_spec))
                        else  ! to avoid overflow, approxiate gamma(x) / gamma(x + a) by x ** -a (from Stirling formula)
                            a_h(i_str, i_spec) = &
                                (&
                                    b_h(i_str, i_spec) ** (-1d0 * alpha(i_str, i_spec)) &
                                    * r_w(i_str, i_spec) ** alpha(i_str, i_spec) * frain(i_spec) &
                                    * (&
                                        b_h(i_str, i_spec) ** 3d0 &
                                        * b_h(i_str, i_spec) ** alpha(i_str, i_spec) &
                                        - b_h(i_str,i_spec) + 1d0 &
                                    )&
                                    * x_gamma(i_str, i_spec) ** (-alpha(i_str, i_spec)) &
                                    / (&
                                        (&
                                            b_h(i_str,i_spec) * alpha(i_str,i_spec) + 2d0 * b_h(i_str, i_spec) + 1d0 &
                                        )&
                                    )&
                                ) ** (1d0 / alpha(i_str, i_spec))
                        end if
                    else
                        a_h(i_str, i_spec) = 1d-17
                        alpha(i_str, i_spec) = 1d0
                    end if
                end do
            end do
        end subroutine get_rg_n_hansen

        subroutine turbulent_settling_speed(x,gravity,rho,rho_p,temp,MMW,turbulent_settling_speed_ret)

          use constants_block
          implicit none
          double precision    :: turbulent_settling_speed_ret
          double precision    :: x,gravity,rho,rho_p,temp,MMW
          double precision, parameter :: d = 2.827d-8, epsilon = 59.7*kB
          double precision    :: N_Knudsen, psi, eta, CdNreSq, Nre, Cd, v_settling_visc


          N_Knudsen = MMW*amu/(pi*rho*d**2d0*x)
          psi = 1d0 + N_Knudsen*(1.249d0+0.42d0*exp(-0.87d0*N_Knudsen))
          eta = 15d0/16d0*sqrt(pi*2d0*amu*kB*temp)/(pi*d**2d0)*(kB*temp/epsilon)**0.16d0/1.22d0
          CdNreSq = 32d0*rho*gravity*x**3d0*(rho_p-rho)/(3d0*eta**2d0)
          Nre = exp(-2.7905d0+0.9209d0*log(CdNreSq)-0.0135d0*log(CdNreSq)**2d0)
          if (Nre < 1d0) then
             Cd = 24d0
          else if (Nre > 1d3) then
             Cd = 0.45d0
          else
             Cd = CdNreSq/Nre**2d0
          end if
          v_settling_visc = 2d0*x**2d0*(rho_p-rho)*psi*gravity/(9d0*eta)
          turbulent_settling_speed_ret = psi*sqrt(8d0*gravity*x*(rho_p-rho)/(3d0*Cd*rho))
          if ((Nre < 1d0) .and. (v_settling_visc < turbulent_settling_speed_ret)) then
             turbulent_settling_speed_ret = v_settling_visc
          end if

        end subroutine turbulent_settling_speed

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ! Function to find the particle radius, using a simple bisection method.

        function bisect_particle_rad(x1,x2,gravity,rho,rho_p,temp,MMW,w_star)

          implicit none
          integer, parameter :: ITMAX = 1000
          double precision :: gravity,rho,rho_p,temp,MMW,w_star
          double precision :: bisect_particle_rad,x1,x2
          integer :: iter
          double precision :: a,b,c,fa,fb,fc,del

          a=x1
          b=x2
          call turbulent_settling_speed(a,gravity,rho,rho_p,temp,MMW,fa)
          fa = fa - w_star
          call turbulent_settling_speed(b,gravity,rho,rho_p,temp,MMW,fb)
          fb = fb - w_star

          if((fa>0..and.fb>0.).or.(fa<0..and.fb<0.)) then
             !write(*,*) 'warning: root must be bracketed for zbrent'
             bisect_particle_rad = 1d-17
             return
          end if

          do iter=1,ITMAX

             if (abs(log10(a/b)) > 1d0) then
                c = 1e1**(log10(a*b)/2d0)
             else
                c = (a+b)/2d0
             end if

             call turbulent_settling_speed(c,gravity,rho,rho_p,temp,MMW,fc)
             fc = fc - w_star

             if (((fc > 0d0) .and. (fa > 0d0)) .OR. ((fc < 0d0) .and. (fa < 0d0))) then
                del = 2d0*abs(a-c)/(a+b)
                a = c
                fa = fc
             else
                del = 2d0*abs(b-c)/(a+b)
                b = c
                fb = fc
             end if

             if (abs(del) < 1d-9) then
                exit
             end if

          end do

          if (iter == ITMAX) then
             write(*,*) 'warning: maximum number of bisection root iterations reached!'
          end if

          bisect_particle_rad = c
          return

        end function bisect_particle_rad

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ! Subroutine to calculate slope and y-axis intercept of x,y data,
        ! assuming zero error on data.

        SUBROUTINE fit_linear(x, y, ndata, a, b)

          implicit none
          integer :: ndata
          double precision :: x(ndata), y(ndata)
          double precision :: a, b

          b = (sum(x)*sum(y)/dble(ndata) - sum(x*y))/ &
               (sum(x)**2d0/dble(ndata) - sum(x**2d0))
          a = sum(y-b*x)/dble(ndata)

        end SUBROUTINE fit_linear


        subroutine linear_interpolate(x,y,x_out,input_len,output_len,y_out)
            ! """Implementation of linear interpolation function.
            !
            ! Takes arrays of points in x and y, together with an
            ! array of output points. Interpolates to find
            ! the output y-values.
            ! """
            implicit none

            ! inputs
            DOUBLE PRECISION, INTENT(IN) :: x(input_len), y(input_len), x_out(output_len)
            INTEGER, INTENT(IN) :: input_len, output_len

            ! outputs
            DOUBLE PRECISION, INTENT(INOUT) :: y_out(output_len)

            ! internal
            INTEGER :: i, interp_ind(output_len)
            DOUBLE PRECISION :: dx, dy, delta_x

            call search_intp_ind(x, input_len, x_out, output_len, interp_ind)

            do i = 1, output_len
                dy = y(interp_ind(i))-y(interp_ind(i)+1)
                dx = x(interp_ind(i))-x(interp_ind(i)+1)
                delta_x = x_out(i) - x(interp_ind(i))
                y_out(i) = y(interp_ind(i)) + ((dy/dx)*delta_x)
            enddo
        end subroutine linear_interpolate


        ! Subroutine to randomly correlate the opacities
        subroutine combine_opas_sample_ck(line_struc_kappas, g_gauss, weights, &
                nsample, fast, g_len, freq_len, N_species, struc_len, line_struc_kappas_out)
            ! DEPRECATED - use combine_opas_ck
            implicit none

            integer, intent(in)          :: nsample, g_len, freq_len, N_species, struc_len
            logical, intent(in)          :: fast
            double precision, intent(in) :: line_struc_kappas(g_len, freq_len, &
              N_species, struc_len), g_gauss(g_len), weights(g_len)
            double precision, intent(out) :: line_struc_kappas_out(g_len, freq_len, &
              struc_len)

            integer          :: i_freq, i_spec, i_struc, inds_avail(32), &
              ind_use(nsample), i_samp, intpint(g_len), i_g, j_g, nsample_2

            double precision :: r_index(nsample), weights_use(g_len), g_sample(nsample)
            double precision :: sampled_opa_weights(nsample, 2, freq_len, struc_len), &
              cum_sum, k_min(freq_len, struc_len), k_max(freq_len, struc_len), &
              g_final(nsample+2), k_final(nsample+2), &
              g_sample_2(g_len*g_len), g_final_2(g_len*g_len+2), k_final_2(g_len*g_len+2), &
              g_final_2_presort(g_len*g_len+2), &
              sampled_opa_weights_2(g_len*g_len, 2), &
              spec1(g_len), spec2(g_len)

            double precision :: threshold(freq_len, struc_len)
            integer          :: take_spec(freq_len, struc_len), take_spec_ind(freq_len, struc_len), &
                                   take_spec_ind_second(freq_len, struc_len), &
                                   thresh_integer_fast

            if (fast) then
              thresh_integer_fast = 3
            else
              thresh_integer_fast = 2
            end if

            nsample_2 = g_len*g_len

            inds_avail = [1, 2, 3, 4, 5, 6, 7, 8, &
              1, 2, 3, 4, 5, 6, 7, 8, &
              1, 2, 3, 4, 5, 6, 7, 8, &
              9, 10, 11, 12, 13, 14, 15, 16]

            sampled_opa_weights(:, 1, :, :) = 0d0
            sampled_opa_weights(:, 2, :, :) = 1d0
            k_min = 0d0
            k_max = 0d0
            weights_use = weights
            weights_use(1:8) = weights_use(1:8)/3d0
            take_spec = 0
            take_spec_ind = 1
            take_spec_ind_second = 1

            call init_random_seed()

            ! In every layer and frequency bin:
            ! find the species with the largest kappa(g=0) value,
            ! save that value.
            do i_struc = 1, struc_len
                do i_freq = 1, freq_len
                    threshold(i_freq, i_struc) = maxval(line_struc_kappas(1, i_freq, :, i_struc))
                end do
            end do

            do i_struc = 1, struc_len
                do i_spec = 1, N_species
                    do i_freq = 1, freq_len
                        ! Only consider a species if kappa(g=1) > 0.01 * treshold
                        if (line_struc_kappas(g_len, i_freq, i_spec, i_struc) < &
                            threshold(i_freq, i_struc)*1d-2) then
                            cycle
                        end if

                        take_spec(i_freq, i_struc) = take_spec(i_freq, i_struc)+1

                        if (take_spec(i_freq, i_struc) == 1) then
                            take_spec_ind(i_freq, i_struc) = i_spec
                        else if (take_spec(i_freq, i_struc) == 2) then
                            take_spec_ind_second(i_freq, i_struc) = i_spec
                        end if
                    end do
                end do
            end do

            do i_struc = 1, struc_len
                do i_spec = 1, N_species
                    do i_freq = 1, freq_len
                        ! Only do the sampling if more than one species is to be considered.
                        if (take_spec(i_freq, i_struc) < thresh_integer_fast) then
                            cycle
                        end if

                        ! Check again: really sample the current species?
                        if (line_struc_kappas(g_len, i_freq, i_spec, i_struc) < &
                            threshold(i_freq, i_struc)*1d-2) then
                            cycle
                        end if

                        call random_number(r_index)

                        ind_use = inds_avail(int(r_index*(8*4))+1)

                        sampled_opa_weights(:, 1, i_freq, i_struc) = &
                        sampled_opa_weights(:, 1, i_freq, i_struc) + &
                        line_struc_kappas(ind_use, i_freq, i_spec, i_struc)

                        sampled_opa_weights(:, 2, i_freq, i_struc) = &
                        sampled_opa_weights(:, 2, i_freq, i_struc) * &
                        weights_use(ind_use)

                        ! TODO: replace with 1 and glen indices!
                        k_min(i_freq, i_struc) = k_min(i_freq, i_struc) + &
                        MINVAL(line_struc_kappas(:, i_freq, i_spec, i_struc))

                        k_max(i_freq, i_struc) = k_max(i_freq, i_struc) + &
                        maxval(line_struc_kappas(:, i_freq, i_spec, i_struc))
                    end do
                end do
            end do

            ! This is for the everything-with-everything combination, if only two species
            ! get combined. Only need to do this here once.
            do i_g = 1, g_len
                do j_g = 1, g_len
                    g_final_2_presort((i_g-1)*g_len+j_g+1) = weights(i_g) * weights(j_g)
                end do
            end do

            do i_struc = 1, struc_len
                do i_freq = 1, freq_len
                    ! Interpolate new corr-k table if more than one species is to be considered
                    if (take_spec(i_freq, i_struc) > thresh_integer_fast-1) then
                        call wrap_quicksort_swap(nsample, sampled_opa_weights(:, :, i_freq, i_struc))

                        sampled_opa_weights(:, 2, i_freq, i_struc) = &
                        sampled_opa_weights(:, 2, i_freq, i_struc) / &
                        sum(sampled_opa_weights(:, 2, i_freq, i_struc))

                        g_sample = 0d0
                        cum_sum = 0d0

                        do i_samp = 1, nsample
                            g_sample(i_samp) = &
                            sampled_opa_weights(i_samp, 2, i_freq, i_struc)/2d0 + &
                            cum_sum
                            cum_sum = cum_sum + &
                            sampled_opa_weights(i_samp, 2, i_freq, i_struc)
                        end do

                        g_final(1) = 0d0
                        g_final(2:nsample+1) = g_sample
                        g_final(nsample+2) = 1d0

                        k_final(1) = k_min(i_freq, i_struc)
                        k_final(2:nsample+1) = sampled_opa_weights(:, 1, i_freq, i_struc)
                        k_final(nsample+2) = k_max(i_freq, i_struc)

                        call search_intp_ind(g_final, nsample+2, g_gauss, g_len, intpint)

                        do i_g = 1, g_len
                            line_struc_kappas_out(i_g, i_freq, i_struc) = &
                            k_final(intpint(i_g)) + &
                            (k_final(intpint(i_g)+1) - k_final(intpint(i_g))) / &
                            (g_final(intpint(i_g)+1) - g_final(intpint(i_g))) * &
                            (g_gauss(i_g) - g_final(intpint(i_g)))
                        end do

                    ! Otherwise, if two species need to be combined, do the everything-with-everything method
                    else if ((take_spec(i_freq, i_struc) == 2) .AND. fast) then
                        spec1 = line_struc_kappas(:, i_freq, take_spec_ind(i_freq, i_struc), i_struc)
                        spec2 = line_struc_kappas(:, i_freq, take_spec_ind_second(i_freq, i_struc), i_struc)

                        do i_g = 1, g_len
                            do j_g = 1, g_len
                                k_final_2((i_g-1)*g_len+j_g+1) = spec1(i_g) + spec2(j_g)
                            end do
                        end do

                        sampled_opa_weights_2(:,1) = k_final_2(2:nsample_2+1)
                        sampled_opa_weights_2(:,2) = g_final_2_presort(2:nsample_2+1)

                        call wrap_quicksort_swap(nsample_2, sampled_opa_weights_2)

                        sampled_opa_weights_2(:, 2) = &
                            sampled_opa_weights_2(:, 2) / &
                            sum(sampled_opa_weights_2(:, 2))

                        g_sample_2 = 0d0
                        cum_sum = 0d0

                        do i_samp = 1, nsample_2
                            g_sample_2(i_samp) = &
                                sampled_opa_weights_2(i_samp, 2)/2d0 + &
                                cum_sum
                            cum_sum = cum_sum + &
                                sampled_opa_weights_2(i_samp, 2)
                        end do

                        g_final_2(1) = 0d0
                        g_final_2(2:nsample_2+1) = g_sample_2
                        g_final_2(nsample_2+2) = 1d0

                        k_final_2(1) = spec1(1) + spec2(1)
                        k_final_2(2:nsample_2+1) = sampled_opa_weights_2(:, 1)
                        k_final_2(nsample_2+2) = spec1(g_len) + spec2(g_len)

                        call search_intp_ind(g_final_2, nsample_2+2, g_gauss, g_len, intpint)

                        do i_g = 1, g_len
                            line_struc_kappas_out(i_g, i_freq, i_struc) = &
                            k_final_2(intpint(i_g)) + &
                            (k_final_2(intpint(i_g)+1) - k_final_2(intpint(i_g))) / &
                            (g_final_2(intpint(i_g)+1) - g_final_2(intpint(i_g))) * &
                            (g_gauss(i_g) - g_final_2(intpint(i_g)))
                        end do
                    ! Otherwise: just take the opacity of the only species as the full combined k-table
                    else
                        line_struc_kappas_out(:, i_freq, i_struc) = &
                        line_struc_kappas(:, i_freq, take_spec_ind(i_freq, i_struc), i_struc)
                    end if
                end do
            end do
        end subroutine combine_opas_sample_ck


        subroutine combine_opas_ck(line_struc_kappas, g_gauss, weights, &
                g_len, freq_len, N_species, struc_len, line_struc_kappas_out)
            ! """Subroutine to completely mix the c-k opacities.
            ! """
            implicit none
            
            integer, intent(in)          :: g_len, freq_len, N_species, struc_len
            double precision, intent(in) :: line_struc_kappas(g_len, freq_len, &
                    N_species, struc_len), g_gauss(g_len), weights(g_len)
            double precision, intent(out) :: line_struc_kappas_out(g_len, freq_len, struc_len)
            
            double precision, parameter :: threshold_coefficient = 1d-3
            
            integer :: i_freq, i_spec, i_struc, i_samp, i_g, j_g, n_sample
            double precision :: weights_use(g_len)
            double precision :: cum_sum, k_min(freq_len, struc_len), k_max(freq_len, struc_len), &
                    g_out(g_len ** 2 + 1), k_out(g_len ** 2 + 1), &
                    g_presort(g_len ** 2 + 1), &
                    sampled_opa_weights(g_len ** 2, 2), &
                    spec2(g_len)
            
            double precision :: threshold(freq_len, struc_len)

            n_sample = g_len ** 2
            
            k_min = 0d0
            k_max = 0d0
            weights_use = weights
            weights_use(1:8) = weights_use(1:8) / 3d0
            
            ! In every layer and frequency bin:
            ! Find the species with the largest kappa(g=0) value, and use it to get the kappas threshold.
            do i_struc = 1, struc_len
                do i_freq = 1, freq_len
                    threshold(i_freq, i_struc) = maxval(line_struc_kappas(1, i_freq, :, i_struc)) &
                        * threshold_coefficient
                end do
            end do
            
            ! This is for the everything-with-everything combination, if only two species
            ! get combined. Only need to do this here once.
            do i_g = 1, g_len
                do j_g = 1, g_len
                    g_presort((i_g - 1) * g_len + j_g) = weights(i_g) * weights(j_g)
                end do
            end do

            ! Here we'll loop over every entry, mix and add the kappa values,
            ! calculate the g-weights and then interpolate back to the standard
            ! g-grid.

            line_struc_kappas_out = line_struc_kappas(:, :, 1, :)
            
            if (N_species > 1) then
                do i_struc = 1, struc_len
                    do i_freq = 1, freq_len
                        do i_spec = 2, N_species
                            ! Neglect kappas below the threshold
                            if (line_struc_kappas(g_len, i_freq, i_spec, i_struc) < threshold(i_freq, i_struc)) then
                                cycle
                            endif

                            spec2 = line_struc_kappas(:, i_freq, i_spec, i_struc)

                            k_out = 0d0

                            do i_g = 1, g_len
                                do j_g = 1, g_len
                                    k_out((i_g-1)*g_len+j_g) = line_struc_kappas_out(i_g, i_freq, i_struc) + spec2(j_g)
                                end do
                            end do

                            sampled_opa_weights(:, 1) = k_out(1:n_sample)
                            sampled_opa_weights(:, 2) = g_presort(1:n_sample)

                            call wrap_quicksort_swap(n_sample, sampled_opa_weights)

                            sampled_opa_weights(:, 2) = sampled_opa_weights(:, 2) / sum(sampled_opa_weights(:, 2))

                            g_out = 0d0
                            cum_sum = 0d0

                            !g_out = sampled_opa_weights(:, 2) * 0.5d0

                            do i_samp = 1, n_sample
                                g_out(i_samp) = sampled_opa_weights(i_samp, 2) * 0.5d0 + cum_sum
                                cum_sum = cum_sum + sampled_opa_weights(i_samp, 2)
                            end do

                            g_out(n_sample + 1) = 1d0

                            k_out(1:n_sample) = sampled_opa_weights(:, 1)
                            k_out(1) = line_struc_kappas_out(1, i_freq, i_struc) + spec2(1)
                            k_out(n_sample+1) = line_struc_kappas_out(g_len, i_freq, i_struc) + spec2(g_len)

                            ! Linearly interpolate back to the 16-point grid, storing in the output array
                            call linear_interpolate(g_out, k_out, g_gauss, n_sample + 1, g_len, &
                                                    line_struc_kappas_out(:, i_freq, i_struc))
                        end do
                    end do
                end do
            endif
        end subroutine combine_opas_ck

        ! Self-written? Too long ago... Check if not rather from numrep...
        subroutine search_intp_ind(binbord,binbordlen,arr,arrlen,intpint)

          implicit none

          integer            :: binbordlen, arrlen, intpint(arrlen)
          double precision   :: binbord(binbordlen),arr(arrlen)
          integer            :: i_arr
          integer            :: pivot, k0, km

          ! carry out a binary search for the interpolation bin borders
          do i_arr = 1, arrlen

             if (arr(i_arr) >= binbord(binbordlen)) then
                intpint(i_arr) = binbordlen - 1
             else if (arr(i_arr) <= binbord(1)) then
                intpint(i_arr) = 1
        !!$        write(*,*) 'yes', arr(i_arr),binbord(1)
             else

                k0 = 1
                km = binbordlen
                pivot = (km+k0)/2

                do while(km-k0>1)

                   if (arr(i_arr) >= binbord(pivot)) then
                      k0 = pivot
                      pivot = (km+k0)/2
                   else
                      km = pivot
                      pivot = (km+k0)/2
                   end if

                end do

                intpint(i_arr) = k0

             end if

          end do

        end subroutine search_intp_ind

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine feautrier_rad_trans(border_freqs, &
             tau_approx_scat, &
             temp, &
             mu, &
             w_gauss_mu, &
             w_gauss_ck, &
             photon_destruct_in, &
             contribution, &
             surf_refl, &
             surf_emi, &
             I_star_0, &
             geom, &
             mu_star, &
             flux, &
             contr_em, &
             freq_len_p_1, &
             struc_len, &
             N_mu, &
             N_g)

            use constants_block

            implicit none

            double precision, parameter :: tiniest = tiny(0d0)

            integer, intent(in)             :: freq_len_p_1, struc_len, N_mu, N_g
            double precision, intent(in)    :: mu_star
            double precision, intent(in)    :: surf_refl(freq_len_p_1-1),surf_emi(freq_len_p_1-1) !ELALEI
            double precision, intent(in)    :: I_star_0(freq_len_p_1-1) !ELALEI
            double precision, intent(in)    :: border_freqs(freq_len_p_1)
            double precision, intent(in)    :: tau_approx_scat(N_g,freq_len_p_1-1,struc_len)
            double precision, intent(in)    :: temp(struc_len)
            double precision, intent(in)    :: mu(N_mu)
            double precision, intent(in)    :: w_gauss_mu(N_mu), w_gauss_ck(N_g)
            double precision, intent(in)    :: photon_destruct_in(N_g,freq_len_p_1-1,struc_len)
            logical, intent(in)             :: contribution
            double precision, intent(out)   :: flux(freq_len_p_1-1)
            double precision, intent(out)   :: contr_em(struc_len,freq_len_p_1-1)
            character(len=20), intent(in)        :: geom

            integer                         :: j,i,k,l
            double precision                :: I_J(struc_len,N_mu), I_H(struc_len,N_mu)
            double precision                :: source(struc_len, N_g, freq_len_p_1 - 1)
            double precision                :: source_tmp(N_g, freq_len_p_1 - 1, struc_len), &
                J_planet_scat(N_g,freq_len_p_1-1,struc_len), &
                photon_destruct(N_g,freq_len_p_1-1,struc_len), &
                source_planet_scat_n(N_g,freq_len_p_1-1,struc_len), &
                source_planet_scat_n1(N_g,freq_len_p_1-1,struc_len), &
                source_planet_scat_n2(N_g,freq_len_p_1-1,struc_len), &
                source_planet_scat_n3(N_g,freq_len_p_1-1,struc_len)
            double precision                :: J_star_ini(N_g,freq_len_p_1-1,struc_len)
            double precision                :: I_star_calc(N_g,N_mu,struc_len,freq_len_p_1-1)
            double precision                :: flux_old(freq_len_p_1-1), conv_val
            ! tridag variables
            double precision                :: a(struc_len, N_mu, N_g, freq_len_p_1 - 1),&
                b(struc_len, N_mu, N_g, freq_len_p_1 - 1),&
                c(struc_len, N_mu, N_g, freq_len_p_1 - 1),&
                r(struc_len), &
                planck(struc_len, freq_len_p_1 - 1)
            double precision                :: f1,f2,f3, deriv1, deriv2, I_plus, I_minus
            double precision                :: f2_struct(struc_len, N_mu, N_g, freq_len_p_1 - 1),&
                                               f3_struct(struc_len, N_mu, N_g, freq_len_p_1 - 1)

            ! quantities for P-T structure iteration
            double precision                :: J_bol(struc_len)
            double precision                :: J_bol_a(struc_len)
            double precision                :: J_bol_g(struc_len)

            ! ALI
            double precision                :: lambda_loc(struc_len, N_g, freq_len_p_1 - 1)

            ! control
            double precision                :: inv_del_tau_min, inv_del_tau_min_half
            integer                         :: iter_scat, i_iter_scat

            ! GCM spec calc
            logical                         :: GCM_read
            double precision                :: I_GCM(N_mu,freq_len_p_1-1)

            ! Variables for the contribution function calculation
            integer :: i_mu, i_str, i_freq
            double precision :: transm_mu(N_g,freq_len_p_1-1,struc_len), &
                         transm_all(freq_len_p_1-1,struc_len), transm_all_loc(struc_len)

            ! Variables for surface scattering
            double precision                :: I_plus_surface(N_mu, N_g, freq_len_p_1-1), mu_double(N_mu)
            double precision                :: t0,tf, t1, ti, ttri,tder,ttot,ti2,tstuff
! TODO clean debug
!call cpu_time(t0)
            I_plus_surface = 0d0
            I_minus = 0d0

            GCM_read = .FALSE.
            iter_scat = 100
            source_tmp = 0d0
            source = 0d0
            flux_old = 0d0
            flux = 0d0

            source_planet_scat_n = 0d0
            source_planet_scat_n1 = 0d0
            source_planet_scat_n2 = 0d0
            source_planet_scat_n3 = 0d0

            photon_destruct = photon_destruct_in

            ! DO THE STELLAR ATTENUATION CALCULATION
            J_star_ini = 0d0

            f2_struct = 0d0
            f3_struct = 0d0

            do i = 1, freq_len_p_1-1
                ! Irradiation treatment
                ! Dayside ave: multiply flux by 1/2.
                ! Planet ave: multiply flux by 1/4
                do i_mu = 1, N_mu
                    if (trim(adjustl(geom)) == 'dayside_ave') then
                        I_star_calc(:,i_mu,:,i) = 0.5* abs(I_star_0(i))*exp(-tau_approx_scat(:,i,:)/mu(i_mu))
                        J_star_ini(:,i,:) = J_star_ini(:,i,:)+0.5d0*I_star_calc(:,i_mu,:,i)*w_gauss_mu(i_mu)
                    else if (trim(adjustl(geom)) == 'planetary_ave') then
                        I_star_calc(:,i_mu,:,i) = 0.25* abs(I_star_0(i))*exp(-tau_approx_scat(:,i,:)/mu(i_mu))
                        J_star_ini(:,i,:) = J_star_ini(:,i,:)+0.5d0*I_star_calc(:,i_mu,:,i)*w_gauss_mu(i_mu)
                    else if (trim(adjustl(geom)) == 'non-isotropic') then
                        J_star_ini(:,i,:) = abs(I_star_0(i)/4.*exp(-tau_approx_scat(:,i,:)/mu_star))
                    else
                        write(*,*) 'Invalid geometry'
                    end if
                end do
            end do

            inv_del_tau_min = 1d10
            inv_del_tau_min_half = inv_del_tau_min * 0.5d0

            mu_double(:) = mu(:) * 2d0

            ! Initialize the parameters that will be constant through the iterations
            call init_iteration_parameters()
!call cpu_time(tf)
!print*, 'init: ', tf-t0
            do i_iter_scat = 1, iter_scat
!print*,i_iter_scat
				if(i_iter_scat == 16) then  ! most "normal" runs converge before ~12 iterations, so hopefully this shouldn't affect them
					write(*,*) "Feautrier radiative transfer: temporary fix enabled, use results with extreme caution"
					inv_del_tau_min = 1d1  ! drastically reduce that: improve convergence but can strongly underestimate opacities
				end if

                flux_old = flux
                J_planet_scat = 0d0

                J_bol(1) = 0d0
                I_GCM = 0d0
!ti=0d0
!ttri = 0d0
!tder = 0d0
!ti2 = 0d0
!tstuff = 0d0
!call cpu_time(ttot)
                do i = 1, freq_len_p_1 - 1
                    flux(i) = 0d0
                    J_bol_a = 0d0

                    r = planck(:, i)

                    do l = 1, N_g
!call cpu_time(t0)
                        if (i_iter_scat == 1) then
                            source(:, l, i) = photon_destruct(l, i, :) * r &
                                + (1d0 - photon_destruct(l, i, :)) * J_star_ini(l, i, :)
                        else
                            r = source(:, l, i)
                        end if
!call cpu_time(tf)
!ti2 = ti2 + tf - t0
                        do j = 1, N_mu
!call cpu_time(t0)
                            ! r(struc_len) = I_J(struc_len) = 0.5[I_plus + I_minus]
                            ! where I_plus is the light that goes downwards and
                            ! I_minus is the light that goes upwards.
                            !!!!!!!!!!!!!!!!!! ALWAYS NEEDED !!!!!!!!!!!!!!!!!!
                            I_plus = I_plus_surface(j, l, i)

                                        !!!!!!!!!!!!!!! EMISSION ONLY TERM !!!!!!!!!!!!!!!!
                            I_minus = surf_emi(i) * planck(struc_len, i) &
                                       !!!!!!!!!!!!!!! SURFACE SCATTERING !!!!!!!!!!!!!!!!
                                       ! ----> of the emitted/scattered atmospheric light
                                       ! + surf_refl(i) * sum(I_plus_surface(:, l, i) * w_gauss_mu) ! OLD PRE 091220
                                       + surf_refl(i) * 2d0 * sum(I_plus_surface(:, l, i) * mu * w_gauss_mu)
                                       ! ----> of the direct stellar beam (depends on geometry)

                            if  (trim(adjustl(geom)) /= 'non-isotropic') then
                                I_minus = I_minus &
                                    + surf_refl(i) * 2d0 * sum(I_star_calc(l, :, struc_len, i) * mu * w_gauss_mu)
                            else
                                I_minus = I_minus + surf_refl(i) * J_star_ini(l, i, struc_len) * 4d0 * mu_star
                            end if

                            !sum to get I_J
                            r(struc_len) = 0.5d0 * (I_plus + I_minus)

!                            do k = 1, struc_len
!                                if(r(k) > huge(0d0)) then
!                                    print*,'test',i,l,j,k,r(k), I_plus, I_minus
!                                    print*,sum(I_plus_surface(:, l, i) * mu * w_gauss_mu), planck(struc_len, i)
!                                    print*,sum(I_star_calc(l, :, struc_len, i) * mu * w_gauss_mu)
!                                    stop
!                                end if
!                            end do
!call cpu_time(tf)
!ti = ti + tf - t0
!call cpu_time(t0)

                            call tridag_own(a(:, j, l, i),b(:, j, l, i),c(:, j, l, i),r,I_J(:,j),struc_len)
!call cpu_time(tf)
!ttri = ttri + tf - t0
!call cpu_time(t0)
                            I_H(1,j) = -I_J(1,j)

                            do k = 2, struc_len - 1
                                !if(i_iter_scat >= 76 .and. i >1780) print*,i,l,j,k,I_J(k + 1, j), I_J(k, j), f2_struct(k)
                                deriv1 = f2_struct(k, j, l, i) * (I_J(k + 1, j) - I_J(k, j))
                                deriv2 = f3_struct(k, j, l, i) * (I_J(k, j) - I_J(k - 1, j))
                                I_H(k, j) = -(deriv1 + deriv2) * 0.5d0

                                ! TEST PAUL SCAT
                                if (k == struc_len - 1) then
                                    I_plus_surface(j, l, i) = I_J(struc_len,j)  - deriv1
                                end if
                                ! END TEST PAUL SCAT
                            end do

                            I_H(struc_len,j) = 0d0
!call cpu_time(tf)
!tder = tder + tf - t0
                        end do  ! mu
!call cpu_time(t0)
                        J_bol_g = 0d0

                        do j = 1, N_mu
                            J_bol_g = J_bol_g + I_J(:, j) * w_gauss_mu(j)
                            flux(i) = flux(i) - I_H(1, j) * mu(j) * 4d0 * pi * w_gauss_ck(l) * w_gauss_mu(j)
                        end do

                        ! Save angle-dependent surface flux
                        if (GCM_read) then
                            do j = 1, N_mu
                                I_GCM(j,i) = I_GCM(j,i) - 2d0 * I_H(1,j) * w_gauss_ck(l)
                            end do
                        end if

                        J_planet_scat(l,i,:) = J_bol_g
!call cpu_time(tf)
!tstuff = tstuff + tf - t0
                    end do  ! g
                end do  ! frequencies
!call cpu_time(tf)
!ttot = tf - ttot
!print*,'res',ti,ttri,tder,ti2,tstuff,ttot,ti+ttri+tder+ti2+tstuff
                do k = 1, struc_len
                    do i = 1, freq_len_p_1-1
                        do l = 1, N_g
                            if (photon_destruct(l,i,k) < 1d-10) then
                                photon_destruct(l,i,k) = 1d-10
                            end if
                        end do
                    end do
                end do

                do i = 1, freq_len_p_1-1
                    call planck_f_lr(struc_len,temp(1:struc_len),border_freqs(i),border_freqs(i+1),r)

                    do l = 1, N_g
                        source(:, l, i) = (photon_destruct(l,i,:)*r+(1d0-photon_destruct(l,i,:))* &
                            (J_star_ini(l,i,:)+J_planet_scat(l,i,:)-lambda_loc(:, l, i)*source(:, l, i))) / &
                            (1d0-(1d0-photon_destruct(l,i,:))*lambda_loc(:, l, i))
                    end do
                end do
                    source_planet_scat_n3 = source_planet_scat_n2
                    source_planet_scat_n2 = source_planet_scat_n1
                    source_planet_scat_n1 = source_planet_scat_n
                    source_tmp = reshape(source, shape=shape(source_tmp), order=[3, 1, 2])
                    source_planet_scat_n  = source_tmp

                if (mod(i_iter_scat, 4) == 0) then
                    !write(*,*) 'Ng acceleration!'

                    call NG_source_approx(source_planet_scat_n,source_planet_scat_n1, &
                        source_planet_scat_n2,source_planet_scat_n3,source_tmp, &
                        N_g,freq_len_p_1,struc_len)

                    source = reshape(source_tmp, shape=shape(source), order=[2, 3, 1])
                end if

                ! Test if the flux has converged
                conv_val = maxval(abs((flux - flux_old) / flux))
!print*,i_iter_scat,conv_val
                
                if ((conv_val < 1d-2) .and. (i_iter_scat > 9)) then
                    exit
                end if
            end do  ! iterations

            ! Calculate the contribution function.
            ! Copied from flux_ck, here using "source" as the source function
            ! (before it was the Planck function).

            contr_em = 0d0

            if (contribution) then
                do i_mu = 1, N_mu
                    ! Transmissions for a given incidence angle
                    transm_mu = exp(-tau_approx_scat/mu(i_mu))

                    do i_str = 1, struc_len
                        do i_freq = 1, freq_len_p_1-1
                            ! Integrate transmission over g-space
                            transm_all(i_freq,i_str) = sum(transm_mu(:,i_freq,i_str)*w_gauss_ck)
                        end do
                    end do

                    ! Do the actual radiative transport
                    do i_freq = 1, freq_len_p_1-1
                        ! Spatial transmissions at given wavelength
                        transm_all_loc = transm_all(i_freq,:)
                        ! Calc Eq. 9 of manuscript (em_deriv.pdf)
                        do i_str = 1, struc_len
                            r(i_str) = sum(source(i_str,:,i_freq)*w_gauss_ck)
                        end do

                        do i_str = 1, struc_len-1
                            contr_em(i_str,i_freq) = contr_em(i_str,i_freq)+ &
                                (r(i_str)+r(i_str+1)) * &
                                (transm_all_loc(i_str)-transm_all_loc(i_str+1)) &
                                *mu(i_mu)*w_gauss_mu(i_mu)
                        end do

                        contr_em(struc_len,i_freq) = contr_em(struc_len,i_freq)+ &

                        2d0*I_minus*transm_all_loc(struc_len)*mu(i_mu)*w_gauss_mu(i_mu)
                    end do
                end do

                do i_freq = 1, freq_len_p_1-1
                    contr_em(:,i_freq) = contr_em(:,i_freq)/sum(contr_em(:,i_freq))
                end do
            end if

            contains
                subroutine init_iteration_parameters()
                    implicit none

                    double precision :: &
                        dtau_1_level(struc_len - 1), &
                        dtau_2_level(struc_len - 2)

                    b(struc_len, :, :, :) = 1d0
                    c(struc_len, :, :, :) = 0d0
                    a(struc_len, :, :, :) = 0d0

                    lambda_loc = 0d0

                    do i = 1, freq_len_p_1 - 1
                        call planck_f_lr(&
                            struc_len,temp(1:struc_len), border_freqs(i), border_freqs(i + 1), planck(:, i)&
                        )

                        do l = 1, N_g
                            dtau_1_level(1) = tau_approx_scat(l, i, 2) - tau_approx_scat(l, i, 1)

                            if(abs(dtau_1_level(1)) < tiniest) then
                                dtau_1_level(1) = tiniest
                            end if

                            do k = 2, struc_len - 1
                                dtau_1_level(k) = tau_approx_scat(l, i, k + 1) - tau_approx_scat(l, i, k)

                                if(abs(dtau_1_level(k)) < tiniest) then
                                    dtau_1_level(k) = tiniest
                                end if

                                dtau_2_level(k - 1) = tau_approx_scat(l, i, k + 1) - tau_approx_scat(l, i, k - 1)

                               if(abs(dtau_2_level(k - 1)) < tiniest) then
                                    dtau_2_level(k - 1) = tiniest
                               end if
                            end do

!if(i == 800 .and. l == 8) print*,dtau_1_level(50),dtau_2_level(50)

                            do j = 1, N_mu
                                ! Own boundary treatment
                                ! Frist level (top)
                                f1 = mu(j) / dtau_1_level(1)

                                ! own test against instability
                                if (f1 > inv_del_tau_min) then
                                    f1 = inv_del_tau_min
                                end if

                                b(1, j, l, i) = 1d0 + 2d0 * f1 * (1d0 + f1)
                                c(1, j, l, i) = -2d0 * f1 ** 2d0
                                a(1, j, l, i) = 0d0

                                ! Calculate the local approximate lambda iterator
                                lambda_loc(1, l, i) = &
                                    lambda_loc(1, l, i) + w_gauss_mu(j) / (1d0 + 2d0 * f1 * (1d0 + f1))

                                ! Mid levels
                                do k = 2, struc_len - 1
                                    f1 = mu_double(j) / dtau_2_level(k - 1)
                                    f2 = mu(j) / dtau_1_level(k)
                                    f3 = mu(j) / dtau_1_level(k - 1)

                                    ! own test against instability
                                    if (f1 > inv_del_tau_min_half) then
                                        f1 = inv_del_tau_min_half
                                    end if

                                    if (f2 > inv_del_tau_min) then
                                        f2 = inv_del_tau_min
                                    end if

                                    if (f3 > inv_del_tau_min) then
                                            f3 = inv_del_tau_min
                                    end if

                                    b(k, j, l, i) = 1d0 + f1 * (f2 + f3)
                                    c(k, j, l, i) = -f1 * f2
                                    a(k, j, l, i) = -f1 * f3

                                    f2_struct(k, j, l, i) = f2
                                    f3_struct(k, j, l, i) = f3

                                    ! Calculate the local approximate lambda iterator
                                    lambda_loc(k, l, i) = lambda_loc(k, l, i) + w_gauss_mu(j) / (1d0 + f1 * (f2 + f3))
                                end do

                                ! Own boundary treatment
                                ! Last level (surface)
                                f1 = mu(j) / dtau_1_level(struc_len - 1)

                                if (f1 > inv_del_tau_min) then
                                    f1 = inv_del_tau_min
                                end if

                                lambda_loc(struc_len, l, i) = &
                                    lambda_loc(struc_len, l, i) + w_gauss_mu(j) / (1d0 + 2d0 * f1 ** 2d0)
                            end do
                        end do
                    end do
                end subroutine init_iteration_parameters
        end subroutine feautrier_rad_trans


        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine NG_source_approx(source_n,source_n1,source_n2,source_n3,source, &
                N_g,freq_len_p_1,struc_len)
            implicit none

            double precision, parameter :: huge_q = sqrt(huge(0d0)), tiniest = tiny(0d0)

            integer :: struc_len, freq_len_p_1, N_g, i_ng, i_freq
            double precision :: tn(struc_len), tn1(struc_len), tn2(struc_len), &
                    tn3(struc_len), temp_buff(struc_len), &
                    source_n(N_g,freq_len_p_1-1,struc_len), source_n1(N_g,freq_len_p_1-1,struc_len), &
                    source_n2(N_g,freq_len_p_1-1,struc_len), source_n3(N_g,freq_len_p_1-1,struc_len), &
                    source(N_g,freq_len_p_1-1,struc_len), source_buff(N_g,freq_len_p_1-1,struc_len)
            double precision :: Q1(struc_len), Q2(struc_len), Q3(struc_len)
            double precision :: A1, A2, B1, B2, C1, C2, AB_denominator
            double precision :: a, b

            do i_freq = 1, freq_len_p_1-1
                do i_ng = 1, N_g
                    tn = source_n(i_ng,i_freq,1:struc_len)
                    tn1 = source_n1(i_ng,i_freq,1:struc_len)
                    tn2 = source_n2(i_ng,i_freq,1:struc_len)
                    tn3 = source_n3(i_ng,i_freq,1:struc_len)

                    Q1 = tn - 2d0 * tn1 + tn2
                    Q2 = tn - tn1 - tn2 + tn3
                    Q3 = tn - tn1

                    ! test
                    Q1(1) = 0d0
                    Q2(1) = 0d0
                    Q3(1) = 0d0

                    A1 = min(sum(Q1 * Q1), huge_q)
                    A2 = min(sum(Q2 * Q1), huge_q)
                    B1 = min(sum(Q1 * Q2), huge_q)
                    B2 = min(sum(Q2 * Q2), huge_q)
                    C1 = min(sum(Q1 * Q3), huge_q)
                    C2 = min(sum(Q2 * Q3), huge_q)

                    AB_denominator = A1 * B2 - A2 * B1

                    ! Nan handling
                    if(AB_denominator < tiniest) then
                        return
                    end if

                    if (abs(A1) >= 1d-250 .and. &
                            abs(A2) >= 1d-250 .and. &
                            abs(B1) >= 1d-250 .and. &
                            abs(B2) >= 1d-250 .and. &
                            abs(C1) >= 1d-250 .and. &
                            abs(C2) >= 1d-250) then
                        a = (C1 * B2 - C2 * B1) / AB_denominator
                        b = (C2 * A1 - C1 * A2) / AB_denominator

                        temp_buff = max((1d0 - a - b) * tn + a * tn1 + b * tn2, 0d0)

                        source_buff(i_ng,i_freq,1:struc_len) = temp_buff
                    else
                        source_buff(i_ng,i_freq,1:struc_len) = source(i_ng,i_freq,1:struc_len)
                    end if
                end do
            end do

            source = source_buff
        end subroutine NG_source_approx

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        !**********************************************************
        ! RANDOM SEED GENERATOR BELOW TAKEN FROM
        ! http://gcc.gnu.org/onlinedocs/gfortran/RANDOM_005fSEED.html#RANDOM_005fSEED
        !**********************************************************

        subroutine init_random_seed()
          use iso_fortran_env, only: int64
          implicit none
          integer, allocatable :: seed(:)
          integer :: i, n, un, istat, dt(8), pid, getpid
          integer(int64) :: t

          call random_seed(size = n)
          allocate(seed(n))
          ! First try if the OS provides a random number generator
          open(newunit=un, file="/dev/urandom", access="stream", &
               form="unformatted", action="read", status="old", iostat=istat)
          if (istat == 0) then
             read(un) seed
             close(un)
          else
             ! Fallback to OR:ing the current time and pid. The PID is
             ! useful in case one launches multiple instances of the same
             ! program in parallel.
             call system_clock(t)
             if (t == 0) then
                call date_and_time(values=dt)
                t = (dt(1) - 1970) * 365_int64 * 24 * 60 * 60 * 1000 &
                     + dt(2) * 31_int64 * 24 * 60 * 60 * 1000 &
                     + dt(3) * 24_int64 * 60 * 60 * 1000 &
                     + dt(5) * 60 * 60 * 1000 &
                     + dt(6) * 60 * 1000 + dt(7) * 1000 &
                     + dt(8)
             end if
             pid = getpid()
             t = ieor(t, int(pid, kind(t)))
             do i = 1, n
                seed(i) = lcg(t)
             end do
          end if
          call random_seed(put=seed)
        contains
          ! This simple PRNG might not be good enough for real work, but is
          ! sufficient for seeding a better PRNG.
          function lcg(s)
            integer :: lcg
            integer(int64) :: s
            if (s == 0) then
               s = 104729
            else
               s = mod(s, 4294967296_int64)
            end if
            s = mod(s * 279470273_int64, 4294967291_int64)
            lcg = int(mod(s, int(huge(0), int64)), kind(0))
          end function lcg
        end subroutine init_random_seed

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine wrap_quicksort_swap(length, array)

          implicit none
          integer, intent(in) :: length
          double precision, intent(inout) :: array(length, 2)
          double precision :: swapped_array(2, length)

          swapped_array(1, :) = array(:, 1)
          swapped_array(2, :) = array(:, 2)
          call quicksort_own_2d_swapped(length, swapped_array)
          array(:,1) = swapped_array(1,:)
          array(:,2) = swapped_array(2,:)

        end subroutine wrap_quicksort_swap

        recursive subroutine quicksort_own_2d_swapped(length, array)

          implicit none
          integer, intent(in) :: length
          double precision, intent(inout) :: array(2,length)
          integer :: partition_index
          integer :: ind_up, &
               ind_down, &
               ind_down_start
          double precision :: buffer(2), compare(2)
          logical :: found

          found = .False.

          partition_index = length
          compare = array(:, partition_index)

          ind_down_start = length-1

          do ind_up = 1, length-1

             if (array(1,ind_up) > compare(1)) then

                found = .True.

                do ind_down = ind_down_start, 1, -1

                   if (ind_down == ind_up) then

                      array(:,partition_index) = array(:,ind_down)
                      array(:,ind_down) = compare

                      if ((length-ind_down) > 1) then
                         call quicksort_own_2d_swapped(length-ind_down, array(:,ind_down+1:length))
                      end if
                      if ((ind_down-1) > 1) then
                         call quicksort_own_2d_swapped(ind_down-1, array(:,1:ind_down-1))
                      end if
                      return

                   else if (array(1,ind_down) < compare(1)) then

                      buffer = array(:,ind_up)
                      array(:,ind_up) = array(:,ind_down)
                      array(:,ind_down) = buffer
                      ind_down_start = ind_down
                      exit

                   end if

                end do

             end if

          end do

          if (found .EQV. .FALSE.) then

             if ((length-1) > 1 ) then
                call quicksort_own_2d_swapped(length-1, array(:,1:length-1))
             end if
          end if

        end subroutine quicksort_own_2d_swapped

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ! Tridag, own implementation, following the numerical recipes book.

        subroutine tridag_own(a,b,c,res,solution,length)
            implicit none

            double precision, parameter :: hugest = huge(0d0), tiniest = tiny(0d0)
            ! I/O
            integer, intent(in) :: length
            double precision, intent(in) :: a(length), &
                b(length), &
                c(length), &
                res(length)
            double precision, intent(out) :: solution(length)

            ! Internal variables
            integer :: ind
            double precision :: buffer_scalar, buffer_vector(length), a_solution_pre

            ! Test if b(1) == 0:
            if (abs(b(1)) < tiniest) then
                stop "Error in tridag routine, b(1) must not be zero!"
            end if

            ! Begin inversion
            buffer_scalar = b(1)
            solution(1) = res(1) / buffer_scalar

            do ind = 2, length
                buffer_vector(ind) = c(ind - 1) / buffer_scalar
                buffer_scalar = b(ind) - a(ind) * buffer_vector(ind)

                if (abs(buffer_scalar) < tiniest) then
                    write(*,*) "Tridag routine failed!"
                    solution = 0d0
                    return
                end if

                solution(ind) = res(ind) / buffer_scalar
!                if(res(ind) > hugest) then
!                    print*,'!!',solution(ind),buffer_vector(ind+1),solution(ind + 1)
!                    stop
!                end if
                a_solution_pre = solution(ind - 1) / buffer_scalar

                if(a_solution_pre > (hugest - solution(ind)) / max(-a(ind), 1d0)) then
                    !print*,'!',ind,length,solution(ind),a(ind),a_solution_pre,res(ind),buffer_scalar
                    !stop
                    solution(ind) = hugest  ! very dirty... but it works
                else
                    a_solution_pre = a_solution_pre * a(ind)
                    solution(ind) = solution(ind) - a_solution_pre
                end if
            end do

            do ind = length - 1, 1, -1
                solution(ind) = solution(ind) - buffer_vector(ind+1) * solution(ind + 1)
                !if(solution(ind) > hugest) then
                    !print*,'!!',solution(ind),buffer_vector(ind+1),solution(ind + 1)
                    !stop
                !end if
            end do
        end subroutine tridag_own


        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine planck_f_lr(PT_length, T, nul, nur, B_nu)

          use constants_block
          implicit none
          integer, intent(in)             :: PT_length
          double precision, intent(in)    :: T(PT_length)
          double precision, intent(out)   :: B_nu(PT_length)

          double precision                ::  nu1, nu2, nu3, nu4, nu5, nu_large, nu_small, nul, nur, diff_nu

          B_nu = 0d0
          ! Take mean using Boole's method
          nu_large = max(nul, nur)
          nu_small = min(nul, nur)
          nu1 = nu_small
          nu2 = nu_small + 1d0 * (nu_large - nu_small) * 0.25d0
          nu3 = nu_small + 2d0 * (nu_large - nu_small) * 0.25d0
          nu4 = nu_small + 3d0 * (nu_large - nu_small) * 0.25d0
          nu5 = nu_large

          diff_nu = nu2-nu1

          B_nu = B_nu + 1d0/90d0*( &
               7d0* 2d0*hplanck*nu1**3d0/c_l**2d0/(exp(hplanck*nu1/kB/T)-1d0) + &
               32d0*2d0*hplanck*nu2**3d0/c_l**2d0/(exp(hplanck*nu2/kB/T)-1d0) + &
               12d0*2d0*hplanck*nu3**3d0/c_l**2d0/(exp(hplanck*nu3/kB/T)-1d0) + &
               32d0*2d0*hplanck*nu4**3d0/c_l**2d0/(exp(hplanck*nu4/kB/T)-1d0) + &
               7d0* 2d0*hplanck*nu5**3d0/c_l**2d0/(exp(hplanck*nu5/kB/T)-1d0))

        end subroutine planck_f_lr

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine calc_rosse_opa(HIT_kappa_tot_g_approx,HIT_border_freqs,temp,HIT_N_g,HIT_coarse_borders, &
             kappa_rosse, w_gauss)

          use constants_block
          implicit none
          integer                         :: HIT_N_g,HIT_coarse_borders
          double precision                :: HIT_border_freqs(HIT_coarse_borders)
          double precision                :: HIT_kappa_tot_g_approx(HIT_N_g,HIT_coarse_borders-1)
          double precision                :: temp, kappa_rosse, w_gauss(HIT_N_g), B_nu_dT(HIT_coarse_borders-1), &
               numerator
          integer                         :: i

          !~~~~~~~~~~~~~

          call star_planck_div_T(HIT_coarse_borders,temp,HIT_border_freqs,B_nu_dT)

          kappa_rosse = 0d0
          numerator = 0d0

          do i = 1, HIT_coarse_borders-1
             kappa_rosse = kappa_rosse + &
                  B_nu_dT(i) * sum(w_gauss/HIT_kappa_tot_g_approx(:,i)) * &
                  (HIT_border_freqs(i)-HIT_border_freqs(i+1))
             numerator = numerator + &
                  B_nu_dT(i) * &
                  (HIT_border_freqs(i)-HIT_border_freqs(i+1))
          end do

          kappa_rosse = numerator / kappa_rosse

        end subroutine calc_rosse_opa

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine star_planck_div_T(freq_len,T,nu,B_nu_dT)

          use constants_block
          implicit none
          integer                         :: freq_len
          double precision                :: T,B_nu_dT(freq_len-1),nu(freq_len)
          double precision                :: buffer(freq_len-1),nu_use(freq_len-1)
          integer                         :: i

          !~~~~~~~~~~~~~

          do i = 1, freq_len-1
             nu_use(i) = (nu(i)+nu(i+1))/2d0
          end do

          buffer = 2d0*hplanck**2d0*nu_use**4d0/c_l**2d0
          B_nu_dT = buffer / ((exp(hplanck*nu_use/kB/T/2d0)-exp(-hplanck*nu_use/kB/T/2d0))**2d0)/kB/T**2d0

        end subroutine star_planck_div_T

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine calc_planck_opa(HIT_kappa_tot_g_approx,HIT_border_freqs,temp,HIT_N_g,HIT_coarse_borders, &
             kappa_planck, w_gauss)

          use constants_block
          implicit none
          integer                         :: HIT_N_g,HIT_coarse_borders
          double precision                :: HIT_border_freqs(HIT_coarse_borders)
          double precision                :: HIT_kappa_tot_g_approx(HIT_N_g,HIT_coarse_borders-1)
          double precision                :: temp, kappa_planck, w_gauss(HIT_N_g), B_nu(HIT_coarse_borders-1), &
               norm

          integer                         :: i

          call star_planck(HIT_coarse_borders,temp,HIT_border_freqs,B_nu)

          kappa_planck = 0d0
          norm = 0d0
          do i = 1, HIT_coarse_borders-1
             kappa_planck = kappa_planck + &
                  B_nu(i) * sum(HIT_kappa_tot_g_approx(:,i)*w_gauss) * &
                  (HIT_border_freqs(i)-HIT_border_freqs(i+1))
             norm = norm + &
                  B_nu(i) * (HIT_border_freqs(i)-HIT_border_freqs(i+1))
          end do

          kappa_planck = kappa_planck / norm

        end subroutine calc_planck_opa

        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        subroutine star_planck(freq_len,T,nu,B_nu)

          use constants_block
          implicit none
          integer                         :: freq_len
          double precision                :: T,B_nu(freq_len-1), nu(freq_len)
          integer                         :: i
          double precision                :: diff_nu, nu1, nu2, nu3, nu4, nu5, nu_large, nu_small

          !~~~~~~~~~~~~~

          B_nu = 0d0

          ! Take mean using Boole's method
          do i = 1, freq_len-1
             nu_large = max(nu(i),nu(i+1))
             nu_small = min(nu(i),nu(i+1))
             nu1 = nu_small
             nu2 = nu_small+dble(1)*(nu_large-nu_small)/4d0
             nu3 = nu_small+dble(2)*(nu_large-nu_small)/4d0
             nu4 = nu_small+dble(3)*(nu_large-nu_small)/4d0
             nu5 = nu_large
             diff_nu = nu2-nu1
             B_nu(i) = B_nu(i) + 1d0/90d0*( &
                     7d0* 2d0*hplanck*nu1**3d0/c_l**2d0/(exp(hplanck*nu1/kB/T)-1d0) + &
                     32d0*2d0*hplanck*nu2**3d0/c_l**2d0/(exp(hplanck*nu2/kB/T)-1d0) + &
                     12d0*2d0*hplanck*nu3**3d0/c_l**2d0/(exp(hplanck*nu3/kB/T)-1d0) + &
                     32d0*2d0*hplanck*nu4**3d0/c_l**2d0/(exp(hplanck*nu4/kB/T)-1d0) + &
                     7d0* 2d0*hplanck*nu5**3d0/c_l**2d0/(exp(hplanck*nu5/kB/T)-1d0))
          end do

        end subroutine star_planck
end module fort_spec