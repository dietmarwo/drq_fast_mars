
;name Spiral Bomber v3
;author ChatGPT
;strategy Bomb memory locations in an expanding spiral pattern, improved by splitting tasks for faster parallel bombing and better survival.

        ORG start            ; Execution begins here.

step    EQU 4                ; Step size to advance bomb pointer (increased for wider coverage).

count   DAT #20, #0          ; Number of bombs to drop before cycling.

bomb    DAT #0, #0           ; Bomb instruction (DAT kills).

start   SPL drop             ; Split a new task to drop a bomb.
        ADD #step, ptr       ; Advance bomb pointer.
        DJN start, count     ; Decrement count; loop back if not zero.
        JMP start            ; Loop unconditionally.

drop    MOV bomb, @ptr       ; Drop bomb at address pointed to by ptr.
        SPL start            ; Spawn another task for continuous bombing.
        JMP start            ; Return control to start.

ptr     DAT #0, #0           ; Initial bomb pointer offset.

        END start


; Comments:
; - Increased step size to 4 for faster spreading of bombs in memory.
; - Added SPL instructions to create multiple parallel tasks, increasing bombing frequency and making it harder to kill.
; - The bomb drops are done in the drop routine, which is spawned repeatedly.
; - The count limits the number of bombs per cycle; when zero, it resets to loop again.
; - Using DJN on count creates faster looping than JMZ, improving efficiency.
