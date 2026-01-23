
;name Spiral Bomber Improved v8
;author ChatGPT
;strategy Spawns multiple bombers each cycle with improved task control,
;          uses SPL bombing with tighter limits to prevent overextension,
;          offsets bombing locations for wider core coverage,
;          advances target pointer in a tighter spiral with controlled step decrement,
;          uses DJN instead of JMP for efficient looping and task pruning.

        ORG start

step    DAT #4, #0            ; Initial step size for spiral coverage

start   SPL bomb              ; Spawn bomber #1: bombs target and splits tasks
        SPL bomb2             ; Spawn bomber #2: bombs near target offset -1
        SPL bomb3             ; Spawn bomber #3: bombs near target offset +1
        ADD.AB step, target   ; Advance bombing target pointer by step to create spiral
        DJN step, start       ; Decrement step; loop back if not zero, tightening spiral
        SPL start             ; When step reaches zero, spawn new start spl task to maintain pressure

bomb    SPL 0, @target        ; Plant splitter bomb at target
        DJN #3, bomb          ; Limit bombing repeats to control task growth

bomb2   SPL 0, @target-1      ; Plant splitter bomb just before target
        DJN #2, bomb2         ; Limit bombing repeats for control

bomb3   SPL 0, @target+1      ; Plant splitter bomb just after target
        DJN #2, bomb3         ; Limit bombing repeats for control

target  DAT #0, #0            ; Pointer to current bombing target

        END start
