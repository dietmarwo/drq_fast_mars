
;name   Skipper Improved v6
;author ChatGPT
;strategy Rapid bombing using post-increment indirect addressing to bomb and advance pointer atomically.
;         Limits task spawning to one initial bomber, reducing overhead and improving survival.
;         Bombers execute tight loop of bombing then spl to self, creating new tasks only as needed.

        ORG     start

step    EQU     4               ; Step size for bombing every 4 instructions

start   spl     bomber          ; Spawn initial bomber task
        jmp     loop            ; Control loop (main task)

loop    add     #step, target  ; Advance target pointer
        mov     #0, @target    ; Bomb current target
        jmp     loop            ; Repeat bombing

target  dat     #0, #0         ; Target pointer

bomber  mov     bomb_data, >target ; Bomb and post-increment target pointer atomically
        spl     bomber          ; Spawn another bomber for multitasking
        jmp     bomber          ; Loop bomber continuously

bomb_data dat   #0, #step       ; Data used to bomb and advance pointer

        END     start
