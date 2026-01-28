
;name random_init_multi_fixed_mutator
;author ExplicitMutator
;strategy Paper - self-replicating SPL chain
ORG start

        SPL.A   $10, >0
        SPL.X   #0, $3527
        SPL.A   $0, $0
        SPL.A   *3200, $15
        SPL.A   $0, *19
        SPL.A   $0, $0

        SPL.B   $0, $0
        SPL.A   $0, $3413
start   SPL.A   $0, $0
        MOV.I   $7983, @ptr
        ADD.AB  #3373, $ptr
        JMZ.X   <4, @83
        JMP.A   $-2
ptr     DAT.F   #0, #3305
        ; Jump decoy
        DAT.A   #0, #0
        JMP.A   $6
        ; DAT field decoy
        DIV.F   #0, #0
        DAT.F   #0, #12
        CMP.AB   }-10, >4
        DAT.F   #0, #0
        DAT.F   #0, #0
bloop   ADD.AB  #3217, $bomb_inj
bomb_inj DAT.F   #0, #0
        MOV.AB  #0, @bomb_inj
        JMP.A   $bloop
        ; DAT field decoy
        DAT.F   #0, #0
        DAT.F   #0, #7985
        DAT.F   #0, #0
        DAT.F   #0, #0
bomb_inj DAT.F   #0, #0
bloop   ADD.AB  #3313, $bomb_inj
        NOP.X   @1, >1
        MOV.AB  #0, @bomb_inj
        JMP.A   $bloop
        ; Imp decoy
        MOV.I   $0, $1
        MOV.I   $7, $1

END
