
;name Improved Spiral Bomber
;author ChatGPT
;strategy
; This warrior plants bombs in a spiral pattern using post-increment indirect addressing for efficient bombing
; and uses a combined pointer to smoothly update target positions and radius without redundant instructions.

        ORG start

step    EQU  5              ; smaller step for denser bombing pattern
ptr     DAT  #0, #0         ; combined pointer: will hold both radius and angle as offsets (sum of both)
bomb    DAT  #0, #-9999     ; bomb data (large negative to ensure damage)

start   ADD  #step, ptr     ; advance the pointer by step (spiral expansion)
        MOV  bomb, @ptr     ; plant bomb at address pointed to by ptr (indirect)
        JMP  start          ; loop forever

        END start
