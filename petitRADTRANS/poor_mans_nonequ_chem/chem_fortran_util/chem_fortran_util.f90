subroutine read_data(NFEH, NCO, Npress, Ntemp, Nspec, module_path, chem_table)

  implicit none
  ! I/O
  integer, intent(in) :: NFEH, NCO, Npress, Ntemp, Nspec
  character*500, intent(in) :: module_path
  double precision, intent(out) :: chem_table(Nspec, Ntemp, Npress, NCO, NFEH)
  ! Internal
  integer :: i_feh, i_co, i_p, i_t

  chem_table = 0d0

  open(unit=13,file=trim(adjustl(module_path)) // &
     'abundance_files/abunds_python.dat',form='unformatted')

  do i_feh = 1, NFEH
     do i_co = 1, NCO
        do i_p = 1, Npress
           do i_t = 1, Ntemp
              read(13) chem_table(1:Nspec,i_t,i_p,i_co,i_feh)
           end do
        end do
     end do
  end do

  ! Do not powerlaw interpolate MMW and nabla_ad, only the abundances!
  ! Use linear interpolation for the former instead.
  chem_table(1:Nspec-2,:,:,:,:) = log10(chem_table(1:Nspec-2,:,:,:,:))
  

  close(13)

end subroutine read_data

subroutine interpolate(COs_goal, FEHs_goal, temps_goal, pressures_goal, &
     COs_large_int, FEHs_large_int, temps_large_int, pressures_large_int, &
     FEHs, COs, temps, pressures, chem_table, N_goal, NFEH, NCO, Npress, &
     Ntemp, Nspec, abundance_arr)

  implicit none
  ! I/O
  integer, intent(in) :: NFEH, NCO, Npress, Ntemp, Nspec, N_goal
  double precision, intent(in) :: COs_goal(N_goal), FEHs_goal(N_goal), &
       temps_goal(N_goal), pressures_goal(N_goal)
  integer, intent(in) :: COs_large_int(N_goal), FEHs_large_int(N_goal), &
       temps_large_int(N_goal), pressures_large_int(N_goal)
  double precision, intent(in) :: COs(NCO), FEHs(NFEH), &
       temps(Ntemp), pressures(Npress)
  double precision, intent(in) :: chem_table(Nspec, Ntemp, Npress, NCO, NFEH)
  double precision, intent(out) :: abundance_arr(Nspec, N_goal)
  ! internal
  integer :: i_goal, i_p, i_t, i_feh, i_co, i_p_take, i_t_take, i_feh_take, &
       i_co_take, i_spec
  double precision :: intp_bound(Nspec, 2, 2, 2, 2), &
       intp_coords(4,2), intp_bound_m1(Nspec, 2, 2, 2), &
       intp_bound_m2(Nspec, 2, 2), intp_bound_m3(Nspec, 2)

  do i_goal = 1, N_goal

     do i_feh = 1,2

        i_feh_take = FEHs_large_int(i_goal)-mod(i_feh,2)

        do i_co = 1,2

           i_co_take = COs_large_int(i_goal)-mod(i_co,2)

           do i_p = 1,2

              i_p_take = pressures_large_int(i_goal)-mod(i_p,2)

              do i_t = 1,2

                 i_t_take = temps_large_int(i_goal)-mod(i_t,2)

                 intp_bound(1:Nspec, i_t, i_p, i_co, i_feh) = &
                      chem_table(1:Nspec, i_t_take, i_p_take, i_co_take, i_feh_take)

                 intp_coords(1, i_t) = temps(i_t_take)

              end do

              intp_coords(2, i_p) = pressures(i_p_take)

           end do

           intp_coords(3, i_co) = COs(i_co_take)

        end do

        intp_coords(4, i_feh) = FEHs(i_feh_take)

     end do

     ! interpolate to correct FEH
     intp_bound_m1(1:Nspec, 1:2, 1:2, 1:2) = intp_bound(1:Nspec, 1:2, 1:2, 1:2, 1) + &
          (intp_bound(1:Nspec, 1:2, 1:2, 1:2, 2) - intp_bound(1:Nspec, 1:2, 1:2, 1:2, 1)) / &
          (intp_coords(4,2) - intp_coords(4,1)) * (FEHs_goal(i_goal) - intp_coords(4,1))

     ! Interpolate to correct C/O
     intp_bound_m2(1:Nspec, 1:2, 1:2) = intp_bound_m1(1:Nspec, 1:2, 1:2, 1) + &
          (intp_bound_m1(1:Nspec, 1:2, 1:2, 2) - intp_bound_m1(1:Nspec, 1:2, 1:2, 1)) / &
          (intp_coords(3,2) - intp_coords(3,1)) * (COs_goal(i_goal) - intp_coords(3,1))

     ! Interpolate to correct pressure
     intp_bound_m3(1:Nspec, 1:2) = intp_bound_m2(1:Nspec, 1:2, 1) + &
          (intp_bound_m2(1:Nspec, 1:2, 2) - intp_bound_m2(1:Nspec, 1:2, 1)) / &
          (intp_coords(2,2) - intp_coords(2,1)) * (pressures_goal(i_goal) - intp_coords(2,1))

     ! Interpolate to correct temperature
     abundance_arr(1:Nspec,i_goal) = intp_bound_m3(1:Nspec, 1) + &
          (intp_bound_m3(1:Nspec, 2) - intp_bound_m3(1:Nspec, 1)) / &
          (intp_coords(1,2) - intp_coords(1,1)) * (temps_goal(i_goal) - intp_coords(1,1))

      do i_spec = 1, Nspec
          if (ISNAN(abundance_arr(i_spec,i_goal))) then
              abundance_arr(i_spec,i_goal) = -50d0
          end if
      end do

  end do

  abundance_arr(1:Nspec-2,:) = 1d1**abundance_arr(1:Nspec-2,:)

end subroutine interpolate
