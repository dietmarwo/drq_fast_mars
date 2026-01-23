
;name Spiral Bomber Improved v7
;author ChatGPT
;strategy
; Triple split spiral bomber with optimized bombing and step growth.
; Uses B-predecrement indirect unlink bombing for reliability.
; Adjusted order of bombing and step growth for maximum throughput.
; Multiplicative step growth accelerates spiral expansion rapidly.
; Spawns bombing threads before step growth to prioritize attacks.

        ORG start

step    DAT #4, #0             ; Initial step size
growth  DAT #3, #0             ; Growth factor
target  DAT #0, #0             ; Current bombing target
bomb    DAT 0, 0              ; Bomb instruction: kills enemy

start   ADD.A step, target     ; Increment bombing target by current step
        MOV.I bomb, <target    ; Bomb using B-predecrement indirect (unlink bombing)
        SPL start+1            ; Spawn bombing thread (next bombing)
        SPL start+3            ; Spawn extra bombing thread for coverage
        SPL stepup             ; Spawn step growth thread after spawns
        JMP start              ; Loop back to start

stepup  MUL.A growth, step     ; Multiply step by growth to accelerate spiral
        JMP start+1            ; Return after bombing threads have spawned

        END start
