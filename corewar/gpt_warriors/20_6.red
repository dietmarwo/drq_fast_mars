
;name impquattro_improved
;author rodrigo setti, optimized by assistant
;strategy self-copying "IMP"-style bomber, improved with continuous bombing for better offensive performance

org start

start   mov.i   #0, >4          ; initialize execution by copying current instruction ahead
        mov.i   }-1, >-1       ; auto-copy next instruction ahead (postincrement indirect to spread bombing)
        djn.b   $-1, loop      ; decrement B-field of next instruction and loop if not zero

loop    spl     0               ; split to create an additional task, increasing execution speed
        mov.ab  #4, $3         ; reload counter for continuous bombing cycles
        jmp     start          ; loop back to start for continuous bombing and spreading

        end
