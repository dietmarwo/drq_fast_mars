
;name    Cycler Improved 4
;author  ChatGPT
;strategy Spawn three parallel processes plus main loop; each bombs every 4th location using postincrement indirect addressing for efficiency.
;         All processes share a common pointer with post-increment addressing, avoiding collisions and covering all core quickly.

        ORG     start

start   SPL     cycle1         ; Spawn process 1
        SPL     cycle2         ; Spawn process 2
        SPL     cycle3         ; Spawn process 3

        MOV     0, >ptr       ; Bomb current target, post-increment ptr with A-number of instruction at ptr
        JMP     start         ; Main loop to keep cycling and bombing

cycle1  MOV     0, >ptr       ; Process 1 bombs and increments pointer
        JMP     cycle1

cycle2  MOV     0, >ptr       ; Process 2 bombs and increments pointer
        JMP     cycle2

cycle3  MOV     0, >ptr       ; Process 3 bombs and increments pointer
        JMP     cycle3

ptr     DAT     #0             ; Pointer to bombing target using post-increment indirect addressing

        END     start
