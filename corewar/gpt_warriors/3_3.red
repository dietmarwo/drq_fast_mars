
;name Spiral Bomber Optimized v7
;author ChatGPT
;strategy
; Doubles bombing threads for faster coverage.
; Uses pre-decrement indirect addressing to reduce self-modification risk.
; Adds SPL in bomb to maintain multiple bombers for better coverage.
; Removed redundant JMP in bomb to reduce overhead.
; Slightly increased step size for better spread while still dense.
; Changed initial SPL to SPL in start for continuous bombing and spawning.
; Replaced JMP bomb with DJN bomb, running bomb 'step' times for better control and less overhead.
; Changed DJN to DJN.B to correctly decrement B-field step counter and avoid unnecessary decrement on A.
; Added initial clear of target to avoid junk data.

        ORG start

step    EQU 5                 ; Slightly increased step for better coverage spread while maintaining density

target  DAT #0, #0            ; Target pointer

start   MOV #0, target        ; Clear target pointer initially
        SPL bomb              ; Spawn a new bombing thread
        ADD.AB #step, target  ; Increment target pointer by step with AB modifier for dual-field increment
        JMP start             ; Loop to spawn bombers continuously

bomb    MOV.I #0, {target     ; Bomb the target using pre-decrement addressing to reduce self-mod risk
        SPL bomb              ; Split off another bomber thread for faster spreading
        DJN.B step, bomb      ; Decrement step B-field and loop bombing 'step' times before ending thread

        END start
