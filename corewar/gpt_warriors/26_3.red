
;name Spiral Bomber Improved v11
;author ChatGPT
;strategy Enhanced spiral bombing with balanced splitting to prevent task explosion,
;          precise cyclic step decrement for stable spiral, and staggered bombing with offsets.

        ORG start

step    DAT #5, #0            ; Moderate initial step balancing speed and control
initial DAT #5, #0            ; Store initial step value for reset

start   SPL bomb              ; Spawn main bomb task
        ADD.AB step, target   ; Advance bombing target pointer by current step
        DJN step, loop        ; Decrement step and loop if not zero

        MOV initial, step     ; Reset step to initial value after cycle
loop    SPL bomb2             ; Spawn secondary bomb task offset -4
        SPL bomb3             ; Spawn tertiary bomb task offset +4
        SPL start             ; Spawn new spiral loop task for continuous pressure

bomb    SPL 0, @target        ; Bomb target location
        DJN #3, bomb          ; Limit repeats to control spawn rate

bomb2   SPL 0, @target-4      ; Bomb four before target for wider spread
        DJN #2, bomb2         ; Reduced repeats to control tasks

bomb3   SPL 0, @target+4      ; Bomb four after target for wider spread
        DJN #2, bomb3         ; Reduced repeats to control tasks

target  DAT #0, #0            ; Pointer to current target address

        END start
