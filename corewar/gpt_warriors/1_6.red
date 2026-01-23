
;name    Cycler Optimized
;author  ChatGPT
;strategy Spawns 4 parallel bombing processes with efficient pointer increments using post-increment indirect addressing.
;         Uses SPL with immediate addressing for fast spawning and avoids redundant infinite loops by letting SPL manage processes.

        ORG     start

start   SPL     #cycle1        ; Spawn process 1
        SPL     #cycle2        ; Spawn process 2
        SPL     #cycle3        ; Spawn process 3
        SPL     #cycle4        ; Spawn process 4
        MOV     0, >ptr        ; Bomb current target and post-increment pointer
        MOV     #0, ptr        ; Reset pointer to zero for cycling bomb locations
        JMP     start          ; Loop back to continue bombing

cycle1  MOV     0, >ptr        ; Bomb and increment pointer
        JMP     cycle1         ; Loop cycle 1

cycle2  MOV     0, >ptr        ; Bomb and increment pointer
        JMP     cycle2         ; Loop cycle 2

cycle3  MOV     0, >ptr        ; Bomb and increment pointer
        JMP     cycle3         ; Loop cycle 3

cycle4  MOV     0, >ptr        ; Bomb and increment pointer
        JMP     cycle4         ; Loop cycle 4

ptr     DAT     #0             ; Pointer for bombing target using post-increment indirect addressing

        END     start
