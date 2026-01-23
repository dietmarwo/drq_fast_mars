
;redcode
;name    Repeater Optimized v18
;author  ChatGPT
;strategy Improved Repeater with staggered bombers and reduced queue usage.
;         Uses SPL only once to spawn all three bombers via chained SPL calls.
;         Uses DJN with predecrement indirect addressing for natural wrap.
;         Bombs placed atomically with MOV.I via B-number indirect addressing.
;         Starts bombing at staggered offsets for collision reduction.

        ORG     start

step    EQU     -4              ; Negative step size to decrement pointers backward

start   mov     #0, target      ; Initialize target pointer to 0 for bomber1
        mov     #step, target+1 ; Initialize target pointer to step for bomber2
        mov     #step*2, target+2 ; Initialize target pointer to step*2 for bomber3
        spl     bomber1         ; Spawn bomber1
        jmp     bomber2         ; Jump to bomber2 (spawned by bomber1's spl)
        
bomber1 spl     bomber3         ; Spawn bomber3 (chained spl reducing multiple spl in start)
        djn.i   step, {target  ; Decrement pointer (A-number indirect) at location pointed by target
        mov.i   0, @target     ; Bomb at location pointed by target B-number indirect
        jmp     bomber1        ; Loop

bomber2 djn.i   step, {target+1
        mov.i   0, @target+1
        jmp     bomber2

bomber3 djn.i   step, {target+2
        mov.i   0, @target+2
        jmp     bomber3

target  dat     #0, #0
        dat     #0, #0
        dat     #0, #0

        END     start
