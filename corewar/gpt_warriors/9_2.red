
;redcode
;name    Repeater Optimized v12
;author  ChatGPT
;strategy Three parallel bombers with predecrement indirect addressing and staggered steps to reduce collisions.
;         Uses DJN to decrement pointers and jump for fast looping.
;         Initial targets set relative to self for portability and speed, and B-number clear for safety.
;         Uses SPL only twice to maintain 3 processes, minimizing overhead.

        ORG     start

step    EQU     4               ; Step size

start   spl     bomber2         ; Spawn second bomber
        spl     bomber3         ; Spawn third bomber
        jmp     bomber1         ; Start first bomber immediately

bomber1 djn.i   #step, {target  ; Decrement pointer by step and jump if nonzero
        mov.i   0, @target      ; Bomb at the address pointed by target
        jmp     bomber1         ; Loop bombing

bomber2 djn.i   #8, {target+step ; Decrement pointer by 8 (2*step) starting at offset step
        mov.i   0, @target+step
        jmp     bomber2

bomber3 djn.i   #12, {target+step*2 ; Decrement pointer by 12 (3*step) starting at offset 2*step
        mov.i   0, @target+step*2
        jmp     bomber3

target  dat     #start, #0       ; Pointer initialized to start (A-number) and zero B-number

        END     start
