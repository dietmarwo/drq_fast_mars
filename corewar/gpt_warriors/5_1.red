
;name Spiral Bomber Optimized v2
;author ChatGPT
;strategy Uses a single bombing loop with dual post-increment indirect bombing,
;         saving instructions and increasing bombing speed and coverage.
;         Eliminates redundant JMP by using SPL + MOV combination in bombing.

        ORG start

step    EQU 2                 ; larger step to speed bombing spread

start   SPL bomb              ; spawn bombing process rapidly
        JMP start             ; main loop keeps spawning bombers

bomb    ADD #step, count      ; increment pointer by step
        MOV.I #0, >count      ; bomb at post-increment indirect address
        MOV.I #0, >count      ; bomb next address as well to double bombing speed
        JMP bomb              ; loop bombing

count   DAT #0, #-2           ; initialize count pointer two behind to start bombing immediately

        END start
