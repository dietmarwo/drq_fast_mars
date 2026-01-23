
;name  Hunter Optimized v4
;author ChatGPT
;strategy
;   Improved hunter that rapidly replicates imps to increase parallelism,
;   while incrementing target pointer by 4 for wider scan coverage.
;   Bomb task now loops, dropping bombs continuously to exploit multiple targets.
;   Uses SPL to create multiple imps and bombers, increasing pressure on opponent.

        ORG start

step    EQU 4                   ; Increased step size for wider scanning

ptr     DAT 0, 0               ; Pointer to current target

start   SPL imp                 ; Spawn imp loop task
        SPL bomb                ; Spawn bomb task
        ADD.AB #step, ptr      ; Increment pointer by step
        JMP start              ; Loop forever

imp     SPL imp                 ; Recursively spawn new imps for rapid replication
        MOV 0, 1               ; Imp loop: copy current instruction to next
        JMP imp                ; Loop forever

bomb    MOV.F #0, @ptr         ; Bomb target (place DAT 0,0)
        ADD.AB #step, ptr      ; Move to next target
        JMP bomb               ; Loop forever

        END start
