
;name    jumper clear improved
;author  Rodrigo Setti (mutated for better performance)
;strategy Clears core and changes position efficiently

ORG     start

start   SPL     #0               ; Split to increase tasks, improving survival chance
        MOV.I   $-2, $10        ; Copy instruction 2 lines before to 10 lines ahead to clear core
        NOP.B   >-1, >-1        ; Wait for next cycle, prevent early looping
        DJN.B   $-2, #493       ; Decrement and jump to keep loop count

        MOV.AB  #493, $-1       ; Set pointer for bombing every 493 steps
        MOV.AB  #10, $-4        ; Set offset for movement

loop    MOV.I   *-6, @-6        ; Copy instruction indirectly to target address
        NOP.B   }-7, >-7        ; Defensive NOPs with post-increment for unpredictability
        SNE.AB  $-8, #11        ; Skip next if not equal to create delay and avoid fast kills

        JMP.B   $-994, #0       ; Loop to start (move execution back to beginning)
        JMP.B   $-4, #0         ; Jump relative based on offset for position change

END     start
