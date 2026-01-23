
;name Spiral Bomber Enhanced v11
;author ChatGPT
;strategy Improved spiral bomber with tighter spirals and multi-bomber attacks,
;          uses DJN to control bomber lifespan and reduce wasted cycles,
;          staggered bombers for broader but efficient core coverage.

        ORG start

step    DAT #3, #0            ; Smaller step for tighter spiral, more frequent bombing

start   SPL bomb              ; Spawn bomber attacking at target
        SPL bomb2             ; Spawn bomber attacking slightly behind target
        SPL bomb3             ; Spawn bomber attacking slightly ahead of target
        ADD.AB step, target   ; Advance target pointer by step amount
        DJN #20, start        ; Use DJN with immediate decrement for controlled iterations

bomb    SPL 0, @target        ; Bomb at address in target's B-field
        DJN #5, bomb          ; Allow 5 bombing attempts for persistence

bomb2   SPL 0, @target-3      ; Bomb 3 steps behind target for wider coverage
        DJN #4, bomb2         ; 4 iterations for secondary bomber

bomb3   SPL 0, @target+3      ; Bomb 3 steps ahead of target for wider coverage
        DJN #4, bomb3         ; 4 iterations for tertiary bomber

target  DAT #0, #0            ; Pointer to current target address

        END start
