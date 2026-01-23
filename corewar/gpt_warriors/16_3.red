
;name Enhanced Looping Bomber v5
;author ChatGPT
;strategy Continuously bomb every 3rd location using 3 staggered bomber threads.
;          Main loop only spawns initial bombers and then executes NOP for efficiency.
;          Bombers use pre-decrement indirect addressing for fast pointer cycling.
;          Eliminated unnecessary JMP in main thread for efficiency.

        ORG start

step    EQU 3                 ; Step size: bomb every 3rd instruction
target  DAT #0, #0            ; Pointer to current target to bomb

start   SPL bomber            ; Spawn bomber thread 1
        SPL bomber+3          ; Spawn bomber thread 2 for stagger
        SPL bomber+6          ; Spawn bomber thread 3 for stagger
        NOP                   ; Idle main thread for efficiency

bomber  MOV.BA #0, {target    ; Bomb using B-number pre-decrement indirect addressing (decrement target before bombing)
        ADD.B #step, target   ; Advance target pointer by step for next bomb
        SPL bomber            ; Resurrect bomber thread to keep bombing continuously
        JMP bomber            ; Loop bomber thread

        END
