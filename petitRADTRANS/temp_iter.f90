!!$ Natural constants block

module constants_block
  implicit none
  DOUBLE PRECISION,parameter      :: AU = 1.49597871d13, R_sun = 6.955d10, R_jup=6.9911d9
  DOUBLE PRECISION,parameter      :: pi = 3.14159265359d0, sig=5.670372622593201d-5, c_l=2.99792458d10
  DOUBLE PRECISION,parameter      :: G = 6.674d-8, M_jup = 1.89813d30, deg = Pi/1.8d2
  DOUBLE PRECISION,parameter      :: kB=1.3806488d-16, hplanck=6.62606957d-27, amu = 1.66053892d-24
  DOUBLE PRECISION,parameter      :: sneep_ubachs_n = 25.47d18, L0 = 2.68676d19
end module constants_block

subroutine temp_iter(press, temp_in, T_int, gravity, kappa_H, kappa_J, eddington_F, eddington_Psi, H_star, &
     abs_S, gammas, do_conv, i_iter, kappa_P, J_bol, convective, convergence_test, start_conv, eddington_approx, &
     mean_last, struc_len, HIT_coarse_borders, HIT_border_freqs, temp_out, convective_use)

  use constants_block
  implicit none

  ! I/O
  INTEGER, intent(in)             :: HIT_coarse_borders, struc_len
  INTEGER, intent(in)             :: i_iter
  DOUBLE PRECISION, intent(in)    :: HIT_border_freqs(HIT_coarse_borders)
  DOUBLE PRECISION, intent(in)    :: press(struc_len), temp_in(struc_len)
  DOUBLE PRECISION, intent(out)   :: temp_out(struc_len)
  DOUBLE PRECISION, intent(in)    :: kappa_J(struc_len), kappa_H(struc_len)
  DOUBLE PRECISION, intent(in)    :: eddington_F(struc_len), eddington_Psi
  DOUBLE PRECISION, intent(in)    :: T_int, gravity
  DOUBLE PRECISION, intent(in)    :: H_star(HIT_coarse_borders-1,struc_len)
  DOUBLE PRECISION, intent(in)    :: abs_S(struc_len)
  DOUBLE PRECISION, intent(in)    :: gammas(struc_len)
  DOUBLE PRECISION                :: ad_grad(struc_len-1)
  DOUBLE PRECISION, intent(in)    :: kappa_P(struc_len), J_bol(struc_len)
  LOGICAL, intent(in)             :: convective(struc_len), convergence_test, start_conv, &
       eddington_approx, mean_last, do_conv
  LOGICAL, intent(out)            :: convective_use(struc_len)
 
  ! internal
  DOUBLE PRECISION                :: eddington_F_use(struc_len), eddington_Psi_use
  INTEGER                         :: i_str, j_str, i_freq, conv_sum, conv_sum_max
  DOUBLE PRECISION                :: J_VEF(struc_len), H_VEF(struc_len)
  DOUBLE PRECISION                :: H_out, H, J, fJ, fJ_old, H_s, kappa_planck, delta_fJ, &
                                       H_old, H_next, delta_fJ_lower_ord
  DOUBLE PRECISION                :: kappa_rosse, gamma_VEF, f_dps
  DOUBLE PRECISION                :: temp_buff, H_s_top, corr_fac_J_conv(struc_len)
  LOGICAL                         :: nan_temp, conv_change
  DOUBLE PRECISION                :: osc_test, osc_old

  ! control
  LOGICAL                         :: second_order

  eddington_F_use = eddington_F
  eddington_Psi_use = eddington_Psi
  convective_use = convective
  
!!$  ! Test for Na / K condensation oscillations
!!$  ! induced by root finding
!!$  if ((.not. negelect_root) .and. (i_iter > 4)) THEN
!!$     do i_str = 3, struc_len-2
!!$        osc_test = 0d0
!!$        osc_old = 0d0
!!$        do j_str = i_str-2, i_str+1
!!$           osc_test = osc_test + (temp_in(j_str)-temp_in(j_str+1))/ABS(temp_in(j_str)-temp_in(j_str+1))
!!$           if (j_str > i_str-2) then
!!$              if (abs((temp_in(j_str)-temp_in(j_str+1))/ABS(temp_in(j_str)-temp_in(j_str+1)) + osc_old) > 1d-7) then
!!$                 osc_test = 4
!!$                 EXIT
!!$              end if
!!$           end if
!!$           osc_old = (temp_in(j_str)-temp_in(j_str+1))/ABS(temp_in(j_str)-temp_in(j_str+1))
!!$        end do
!!$        if (ABS(osc_test) < 1d-4) then
!!$           negelect_root = .TRUE.
!!$           temp_in = T_int*3d0/4d0
!!$           write(*,*) 'Oscillation detected! zbrent root disabled! Use classical temperature scheme!'
!!$        end if
!!$     end do
!!$  end if

  ! We want to solve the VEF integration using as second order accurate integration
  second_order = .TRUE.

  ! Test Eddington approximation?
  if (eddington_approx) then
     eddington_psi_use = -0.5d0
     eddington_F_use = 1d0/3d0
  end if

  ! Variable setup for convection max. 2 layer criterion
  conv_change = .FALSE.
  conv_sum = 0
  conv_sum_max = 2

  ! Correction factor to allow continued moment integration if convection is turned on
  corr_fac_J_conv = J_bol / (sig*temp_in**4d0/pi)

  ! Calc. nabla_ad from gammas
  do i_str = 1, struc_len-1
     ad_grad(i_str) = ((gammas(i_str)-1d0)/(gammas(i_str))+(gammas(i_str+1)-1d0)/(gammas(i_str+1)))/2d0
  end do

  ! This is the net flux that is lost by the atmosphere (times 4 pi, as H = 4 pi F).
  ! We will ust that this must be conserved to find our temperature solution.
  H_out = sig*T_int**4d0/(4d0*pi)

  !---------------------------------------------
  !---------------------------------------------
  !---------------------------------------------
  !---------------------------------------------
  ! Start temperature solution!
  !---------------------------------------------
  !---------------------------------------------
  !---------------------------------------------
  !---------------------------------------------

  !---------------------------------------------  
  ! Do the outermost point ("P=0")
  !---------------------------------------------

  ! What is the current stellar (downward) flux visible at a given layer?
  ! H_star comes from the full stellar Feautrier solution.
  H_s = 0d0
  do i_freq = 1, HIT_coarse_borders-1
     H_s = H_s + H_star(i_freq,1)*(HIT_border_freqs(i_freq)-HIT_border_freqs(i_freq+1))
  end do

  H_s_top = H_s
  ! H is the planetary flux, using net flux convergence.
  H = H_out - H_s
  ! This gives us the starting boundary condition for the downwards integration,
  ! using the first variable Eddington factor psi
  ! (which is equal to 1/2 in the classical isotropic Eddingtion assumption).
  J = H / abs(eddington_Psi_use)

  ! Saving J and H into J_VEF, H_VEF: this allows comparing with J_bol, H_bol later:
  ! _VEF and _bol should be agreeing in a converged solution.
  J_VEF(1) = J
  H_VEF(1) = H

  ! TODO: to be read in from outside!
!!$  call  calc_planck_opa(HIT_kappa_tot_g_approx(:,:,1),HIT_border_freqs,temp(1),HIT_N_g_eff,HIT_coarse_borders, &
!!$       kappa_planck, w_gauss_ck,HIT_N_species,water_HITRAN,HITRAN_all,planck_grid,CIA_H2_ind,HIT_mfr(:,1), &
!!$       HIT_nfr(:,1),nfrHe(1),rho(1),press(1),kappa_planck_2,i_iter,CIA_inc_H2_H2,CIA_inc_H2_He)

  ! Temperature solution in the uppermost layer, leveraging radiative equilibrium.
  temp_buff = (pi/sig*(kappa_J(1)*J+abs_S(1))/(kappa_P(1)))**(0.25d0)

  ! Check whether that worked alright, or not.
  nan_temp = .FALSE.
  if (temp_buff .NE. temp_buff) then
     write(*,*) 'nan tempbuff',kappa_J(1),J,abs_S(1),kappa_P(1)
     nan_temp = .TRUE.
  end if

  ! Mixing between old and new temperature solution (gives stability).
  ! Keep old temperature if nan problem detected.
  if (mean_last) then
     if (.NOT. nan_temp) then
        if (i_iter < 750) then  ! was 0.9d0       was 0.1d0
           temp_out(1) = (temp_in(1)*0.3d0+temp_buff*0.7d0)
        else
           temp_out(1) = (temp_in(1)*0.95d0+temp_buff*0.05d0)
        end if
     end if
  else
     if (.NOT. nan_temp) then
        temp_out(1) = temp_buff
     end if
  end if

  ! TODO: to be read in from outside!
!!$  call  calc_planck_opa(HIT_kappa_tot_g_approx(:,:,1),HIT_border_freqs,temp(1),HIT_N_g_eff,HIT_coarse_borders, &
!!$       kappa_planck, w_gauss_ck,HIT_N_species,water_HITRAN,HITRAN_all,planck_grid,CIA_H2_ind,HIT_mfr(:,1), &
!!$       HIT_nfr(:,1),nfrHe(1),rho(1),press(1),kappa_planck_2,i_iter,CIA_inc_H2_H2,CIA_inc_H2_He)

  ! TODO: Check if still needed.
!!$  kappa_P(1,1) = kappa_planck
!!$  kappa_P(1,2) = kappa_planck_2

  ! HERE!
  
  !---------------------------------------------
  ! Go down to P > 0!
  !---------------------------------------------

  ! We are actually intergrating d(f*J)/dP in the VEF method, so this is the starting value for the
  ! downwards integration
  fJ = J*eddington_F_use(1)

  do i_str = 2, struc_len

     ! We will add delta_fJ to this value later to integrate fJ downwards
     fJ_old = fJ
     ! This is the planetary flux in the layer above
     H_old = H
     if (i_str .EQ. 2) then
        ! Required H in the current layer
        H_s = 0
        do i_freq = 1, HIT_coarse_borders-1
           H_s = H_s + H_star(i_freq,i_str)*(HIT_border_freqs(i_freq)-HIT_border_freqs(i_freq+1))
        end do
        H = H_out - H_s
        ! Required H in the next layer
        H_s = 0d0
        do i_freq = 1, HIT_coarse_borders-1
           H_s = H_s + H_star(i_freq,i_str+1)*(HIT_border_freqs(i_freq)-HIT_border_freqs(i_freq+1))
        end do
        H_next = H_out - H_s
     else
        if (i_str .EQ. struc_len) then
           ! For the bottom layer there is no next layer
           H = H_next
        else
           ! For layers with i_str > 2 H = H_next, so just calculate the new H_next.
           H = H_next
           H_s = 0d0
           do i_freq = 1, HIT_coarse_borders-1
              H_s = H_s + H_star(i_freq,i_str+1)*(HIT_border_freqs(i_freq)-HIT_border_freqs(i_freq+1))
           end do
           H_next = H_out - H_s
        end if
     end if

     ! Now, lets integrate the equation for d(fJ)/dP using a second order accurate method
     if (second_order) then
        if (i_str < struc_len) then
           f_dps = (press(i_str+1)-press(i_str))/(press(i_str)-press(i_str-1))
           gamma_VEF = (H_next*kappa_H(i_str+1)-(1d0+f_dps)*H*kappa_H(i_str)+ &
                f_dps*kappa_H(i_str-1)*H_old)/f_dps/(1+f_dps)
                                                                 ! second order correction term
           delta_fJ = ((kappa_H(i_str-1)*H_old+kappa_H(i_str)*H)/2d0-gamma_VEF/6d0) * &
                1d0/gravity*(press(i_str)-press(i_str-1))
           ! To ensure numerical stability
           if (delta_fJ < 0d0) then
              delta_fJ = 0d0
           end if
           ! To ensure numerical stability
           delta_fJ_lower_ord = (kappa_H(i_str-1)*H_old+kappa_H(i_str)*H)/2d0/gravity*(press(i_str)-press(i_str-1))
           if (delta_fJ > delta_fJ_lower_ord) then
              delta_fJ = delta_fJ_lower_ord
           end if
        ! There is not 3-point stencil possible for i_str == struc_len
        else
           delta_fJ = (kappa_H(i_str-1)*H_old+kappa_H(i_str)*H)/2d0/gravity*(press(i_str)-press(i_str-1))
        end if
     ! In case we just want to do lower order after all...
     else
        delta_fJ = (kappa_H(i_str-1)*H_old+kappa_H(i_str)*H)/2d0/gravity*(press(i_str)-press(i_str-1))
     end if

     ! Advance fJ for continued downward integration
     fJ = fJ_old + delta_fJ

     ! Now, let transform to J from f*J, using the variable eddinction factor f
     ! measured from the full RT solution (Feautrier).
     J = fJ / eddington_F_use(i_str)

     ! Again, safe for posterity, to be able to check that _VEF = _bol
     J_VEF(i_str) = J
     H_VEF(i_str) = H

     ! J should not be negative...
     if (J < 0d0) then

        write(*,*) 'j_corr', i_str
        ! TODO: replace with kappa_planck being read in from outside. Actually needed here, btw., maybe just for analysis?
!!$        call  calc_planck_opa(HIT_kappa_tot_g_approx(:,:,i_str),HIT_border_freqs,temp(i_str),HIT_N_g_eff,HIT_coarse_borders, &
!!$             kappa_planck, w_gauss_ck,HIT_N_species,water_HITRAN,HITRAN_all,planck_grid,CIA_H2_ind,HIT_mfr(:,i_str), &
!!$             HIT_nfr(:,i_str),nfrHe(i_str),rho(i_str),press(i_str),kappa_planck_2,i_iter,CIA_inc_H2_H2,CIA_inc_H2_He)
!!$        write(*,*) (kappa_H(i_str-1)+kappa_H(i_str))/2d0,kappa_planck,(kappa_H(i_str-1)+kappa_H(i_str))/2d0/kappa_planck
        ! Assume that we are diffusive to get a simpy J correction, using the old temperature.
        J = sig*(temp_in(i_str))**4d0/pi ! 1

     end if

     ! TODO: replace with kappa_planck being read in from outside.
!!$     call  calc_planck_opa(HIT_kappa_tot_g_approx(:,:,i_str),HIT_border_freqs,temp(i_str),HIT_N_g_eff,HIT_coarse_borders, &
!!$          kappa_planck, w_gauss_ck,HIT_N_species,water_HITRAN,HITRAN_all,planck_grid,CIA_H2_ind,HIT_mfr(:,i_str), &
!!$          HIT_nfr(:,i_str),nfrHe(i_str),rho(i_str),press(i_str),kappa_planck_2,i_iter,CIA_inc_H2_H2,CIA_inc_H2_He)

     ! Temperature solution in the current layer, leveraging radiative equilibrium.
     temp_buff = (pi/sig*(kappa_J(i_str)*J+abs_S(i_str))/kappa_P(i_str))**(0.25d0)

     ! Because of thermal inversions H can be smaller than 0 at some spots and which can make J < 0 at some places during
     ! the iteration (not in the converged solution!). Try to make this vanish here...
     ! New comment by Paul, August 2021: but we check for negatuve J above?
     nan_temp = .FALSE.
     if (temp_buff .NE. temp_buff) then
        temp_buff = ((pi/sig*(abs_S(i_str))/kappa_P(i_str))**(0.25d0) + temp_in(i_str))/2d0
        write(*,*) 'nan tempbuff',kappa_J(i_str),J,abs_S(i_str),kappa_P(i_str), pi/sig*(kappa_J(i_str)*J+abs_S(i_str))/ &
             kappa_P(i_str)
        write(*,*) (kappa_H(i_str-1)+kappa_H(i_str))/2d0, i_str
        read(*,*)
        !nan_temp = .TRUE.
     end if

     ! Mixing between old and new temperature solution (gives stability).
     ! Keep old temperature if nan problem detected.
     if (mean_last) then
        if (.NOT. nan_temp) then
           if (i_iter < 750) then
              temp_out(i_str) = (temp_in(i_str)*0.9d0+temp_buff*0.1d0)
           else
              temp_out(i_str) = (temp_in(i_str)*0.95d0+temp_buff*0.05d0)
           end if
        end if
     else
        if (.NOT. nan_temp) then
           temp_out(i_str) = temp_buff
        end if
     end if

     !-------------------------------
     ! CONVECTION
     !-------------------------------

     if (do_conv) then
        if ( (convergence_test .AND. start_conv) .OR. ((.NOT. convergence_test) .AND. i_iter > 50) ) then

           if (press(i_str)*1d-6 > 1d-6) then
              if (((temp_out(i_str)-temp_out(i_str-1))/(press(i_str)-press(i_str-1)) * &
                   (press(i_str)+press(i_str-1))/(temp_out(i_str)+temp_out(i_str-1))) > &
                   ad_grad(i_str-1)) then
                 if (convective_use(i_str) .OR. (.NOT. conv_change)) then
                    write(*,*) i_str, 'conv!'
                    temp_out(i_str) = (temp_out(i_str-1) + ad_grad(i_str-1) * temp_out(i_str-1) / &
                         (press(i_str)+press(i_str-1))*(press(i_str)-press(i_str-1)))/ &
                         (1d0 - ad_grad(i_str-1) * (press(i_str)-press(i_str-1)) / &
                         (press(i_str)+press(i_str-1)))
                    ! fJ = sig/pi*temp_out(i_str)**4d0 * eddington_F_use(i_str)
                    fJ = sig/pi*temp_out(i_str)**4d0*corr_fac_J_conv(i_str) * eddington_F_use(i_str)
                    if (.NOT. convective_use(i_str)) then
                       convective_use(i_str) = .TRUE.
                       conv_sum = conv_sum + 1
                       if (conv_sum == conv_sum_max) then
                          conv_change = .TRUE.
                       end if
                    end if
                 end if
              else
                 if (convective_use(i_str)) then
                    if (conv_change) then
                       write(*,*) i_str, 'conv!'
                       temp_out(i_str) = (temp_out(i_str-1) + ad_grad(i_str-1) * temp_out(i_str-1) / &
                            (press(i_str)+press(i_str-1))*(press(i_str)-press(i_str-1)))/ &
                            (1d0 - ad_grad(i_str-1) * (press(i_str)-press(i_str-1)) / &
                            (press(i_str)+press(i_str-1)))
                       ! fJ = sig/pi*temp_out(i_str)**4d0 * eddington_F_use(i_str)
                       fJ = sig/pi*temp_out(i_str)**4d0*corr_fac_J_conv(i_str) * eddington_F_use(i_str)
                    else
                       conv_sum = conv_sum + 1
                       if (conv_sum == conv_sum_max) then
                          conv_change = .TRUE.
                       end if
                       convective_use(i_str) = .FALSE.
                    end if
                 end if
              end if
           end if

        end if
     end if

     ! TODO: add reading in calc_planck_opa from the outside (also: this may not actually be needed
     ! for the final solution.
!!$     call  calc_planck_opa(HIT_kappa_tot_g_approx(:,:,i_str),HIT_border_freqs,temp(i_str),HIT_N_g_eff,HIT_coarse_borders, &
!!$          kappa_planck, w_gauss_ck,HIT_N_species,water_HITRAN,HITRAN_all,planck_grid,CIA_H2_ind,HIT_mfr(:,i_str), &
!!$          HIT_nfr(:,i_str),nfrHe(i_str),rho(i_str),press(i_str),kappa_planck_2,i_iter,CIA_inc_H2_H2,CIA_inc_H2_He)

!!$     kappa_P(i_str,1) = kappa_planck
!!$     kappa_P(i_str,2) = kappa_planck_2

     !write(125,*) press(i_str)*1d-6, temp(i_str), H, J, abs_S(i_str), eddington_F_use(i_str)

  end do
  
end subroutine temp_iter

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!!$subroutine NG_temp_approx(tn,tn1,tn2,tn3,temp,struc_len)
!!$
!!$  implicit none
!!$  INTEGER :: struc_len, i
!!$  DOUBLE PRECISION :: tn(struc_len), tn1(struc_len), tn2(struc_len), &
!!$       tn3(struc_len), temp(struc_len), temp_buff(struc_len)
!!$  DOUBLE PRECISION :: Q1(struc_len), Q2(struc_len), Q3(struc_len)
!!$  DOUBLE PRECISION :: A1, A2, B1, B2, C1, C2
!!$  DOUBLE PRECISION :: a, b
!!$
!!$  Q1 = tn - 2d0*tn1 + tn2
!!$  Q2 = tn - tn1 - tn2 + tn3
!!$  Q3 = tn - tn1
!!$
!!$  ! test
!!$  Q1(1) = 0d0
!!$  Q2(1) = 0d0
!!$  Q3(1) = 0d0
!!$
!!$  A1 = sum(Q1*Q1)
!!$  A2 = sum(Q2*Q1)
!!$  B1 = sum(Q1*Q2)
!!$  B2 = sum(Q2*Q2)
!!$  C1 = sum(Q1*Q3)
!!$  C2 = sum(Q2*Q3)
!!$
!!$  a = (C1*B2-C2*B1)/(A1*B2-A2*B1)
!!$  b = (C2*A1-C1*A2)/(A1*B2-A2*B1)
!!$
!!$  temp_buff = (1d0-a-b)*tn + a*tn1 + b*tn2
!!$  if (temp_buff(1) <= 0d0) then
!!$     temp_buff(1) = 100d0
!!$  end if
!!$  temp_buff = temp_buff**0.25
!!$
!!$  do i = 1,struc_len
!!$     if (temp_buff(i) .NE. temp_buff(i)) then
!!$        return
!!$     end if
!!$  end do
!!$
!!$  if (sum(tn**0.25d0 - tn1**0.25d0)/sum(temp_buff-temp) > 0d0) then
!!$       temp = temp_buff
!!$  end if
!!$  
!!$end subroutine NG_temp_approx
