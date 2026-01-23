
;name Skipper Improved v12
;author ChatGPT
;strategy Dual-process fast bomber that minimizes overhead and maximizes bombing speed
;          Uses two bombing processes that bomb and post-increment pointer atomically, 
;          and one fast advancing process incrementing pointer by 4 to cover core quickly
;          Removes unnecessary SPL in tight loops for better multitasking balance

        ORG start

step    EQU 4                      ; Step size for spacing

start   MOV.AB  #step, pointer         ; Initialize pointer with step size
        SPL     bomba                 ; Spawn bombing process A
        SPL     bombb                 ; Spawn bombing process B
        JMP     advance               ; Start advancing pointer quickly

bomba   MOV.AB  #0, }pointer         ; Bomb at pointer with post-increment indirect A
        JMP     bomba                 ; Repeat bombing without spawning more processes

bombb   MOV.AB  #0, }pointer         ; Bomb at pointer with post-increment indirect A
        JMP     bombb                 ; Repeat bombing without spawning more processes

advance ADD.AB  #step, pointer         ; Advance pointer by step (4)
        JMP     start                 ; Loop back to start to resync bombing

pointer DAT     #0,     #0            ; Pointer to bombing target

        END start
