
;name random_init_1v1_fixed_mutator
;author ExplicitMutator1v1
;strategy Stone with imp gate defense
ORG start
start   MOV.F   $bomb, @ptr
gate    SPL.A   $0, <gate-4
        ADD.AB  #3, $ptr
bomb    DJN.F   #7990, <0
ptr     JMP.F   {7, #3343
        ; Core clear finishing move
clr     JMP.B   $cdat, >cptr
        JMP.A   $start
cdat    JMZ.BA   #7997, }7995
cptr    JMP.AB   }-1, #3
ccnt    DJN.F   {349, #352

END
