
;name Spiral Bomber
;author ChatGPT
;strategy Bombs memory cells in a spiraling pattern around the starting point,
;           incrementing step each full cycle to cover more core gradually.

        ORG start

step    EQU 5                  ; Step size increment for spiral

start   MOV  bomb, ptr         ; Initialize pointer with bomb address
        MOV  #0, bomb          ; Clear bomb cell (used to hold offsets)

loop    ADD  #step, ptr        ; Increment pointer by step
        MOV  bomb, @ptr        ; Bomb the memory cell pointed by ptr indirectly
        JMP  loop              ; Repeat indefinitely

ptr     DAT  #0, #0            ; Current pointer value
bomb    DAT  #0, #1            ; Bomb dat instruction to write

        END start
