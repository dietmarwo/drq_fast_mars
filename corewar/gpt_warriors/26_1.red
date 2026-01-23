
;name Spiral Bomber Improved v7
;author ChatGPT
;strategy Spawns multiple bombers each cycle at step intervals,
;          uses SPL bombing to rapidly multiply tasks and attack,
;          offsets bombing locations for wider core coverage,
;          advances target pointer in tighter spiral for greater disruption,
;          adds self-splitting to maximize task generation and survivability,
;          improves loop efficiency by replacing JMP with DJN for tighter control and task pruning.

        ORG start

step    DAT #4, #0            ; Increased step size for wider spiral coverage and speed

start   SPL bomb              ; Spawn bomber #1: bombs target and spawns more tasks
        SPL bomb2             ; Spawn bomber #2: bombs near target offset -1
        SPL bomb3             ; Spawn bomber #3: bombs near target offset +1
        ADD.AB step, target   ; Advance bombing target pointer by step to create spiral
        DJN step, start       ; Decrement step each cycle to tighten spiral and loop if step not zero
        SPL start             ; If step is zero, spawn new starting tasks to maintain task pressure

bomb    SPL 0, @target        ; Plant splitter bomb at target, spawn new task at current
        DJN #3, bomb          ; Limit bombing repeats per task to prevent task explosion

bomb2   SPL 0, @target-1      ; Plant splitter bomb one before target
        DJN #2, bomb2         ; Limit bombing repeats per task for control

bomb3   SPL 0, @target+1      ; Plant splitter bomb one after target
        DJN #2, bomb3         ; Limit bombing repeats per task for control

target  DAT #0, #0            ; Pointer to current bombing target

        END start
