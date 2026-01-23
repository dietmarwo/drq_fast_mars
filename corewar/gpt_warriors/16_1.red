
;redcode
;name    Spiral Bomber
;author  ChatGPT
;strategy Continuously bombs memory cells in an increasing spiral pattern to disrupt enemy code.

        ORG start

step    EQU 5                ; Distance to move each step
count   DAT #0, #0           ; Counter for steps taken
pos     DAT #0, #0           ; Current bombing position

start   ADD #step, count     ; Increment step counter by step size
        ADD.I #1, pos        ; Move bombing position forward by 1
        MOV.I #0, @pos       ; Bomb the memory cell at position pointed by pos
        JMZ bomb, count      ; If count mod step == 0, jump to bomb
        JMP start            ; Continue looping

bomb    ADD #step, pos       ; Increase bomb position by step size to spiral out
        JMP start            ; Continue looping

        END start
