
;name   scanner vampire optimized v3
;author rodrigo setti (mutated by assistant)
;strat  aggressive splitting with faster scanning and trap deployment loops

        ORG     start

start   SPL     #3, 0              ; spawn 3 processes for rapid expansion
        MOV.I   $-2, >9           ; clear memory ahead to avoid buildup

scan    JMZ.F   $0, {-4            ; if target is zero, jump back 4 using predecrement indirect (tighter control)
        SLT.A   #16, $-5          ; scan threshold increased to 16 for wider search
        JMP.B   $-4, {-6           ; loop jump back 6 steps for tight cycle

        MOV.A   $-8, $4           ; calculate trap target with proper wrapping
        MUL.A   #-1, $3           ; invert offset for wrap-around handling
        MOV.I   $3, *-9           ; deploy trap by copying instruction robustly

        SPL     #2                ; spawn 2 more processes to cover traps faster
        JMP.B   $-8, {-11          ; loop back 11 steps for continuous scanning/trapping

trapptr JMP.B   $0, 1              ; trap capturer entry point

        END     start
