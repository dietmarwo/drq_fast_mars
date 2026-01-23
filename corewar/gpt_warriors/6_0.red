
;name Mirror Bomber Improved
;author ChatGPT
;strategy A more efficient self-bomber that increments pointer by 4 to bomb every 4th instruction,
;          slowing self-destruction and giving more chances to hit opponents.

        ORG start

start   ADD   #4, ptr       ; Increment pointer by 4 (step size)
        MOV   0, @ptr       ; Bomb the instruction pointed by ptr (indirect bombing)
        JMP   start         ; Repeat bombing loop

ptr     DAT   #0, #0       ; Pointer initialized to zero

        END
