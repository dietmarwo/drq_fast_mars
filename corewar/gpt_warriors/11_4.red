
;name Spiral Bomber Optimized v12
;author ChatGPT
;strategy Optimize task splitting by only splitting when counter > 1 to reduce overhead;
;          combine DJN/JMZ into single DJN check for faster exit paths;
;          initialize counters and targets more compactly;
;          tight main loop removed (not needed), spl tasks handle all bombing.

        ORG start

step    EQU 4                  ; step size for spiral bombing
count   EQU 20                 ; number of bombs per bomber

start   MOV     #count - 1, counter0        ; counters 19 for DJN usage (0-based)
        MOV     #count - 1, counter1
        MOV     #count - 1, counter2
        MOV     #count - 1, counter3

        MOV     counter0, target0            ; initialize targets 0,1,2,3
        MOV     counter0, target1
        ADD     #1, target1
        MOV     counter0, target2
        ADD     #2, target2
        MOV     counter0, target3
        ADD     #3, target3

        SPL     bomber0
        SPL     bomber1
        SPL     bomber2
        SPL     bomber3

bomber0 MOV     bomb, >target0
        ADD     #step, target0
        DJN     bomber0, counter0            ; automatically loop while counter != 0

bomber1 MOV     bomb, >target1
        ADD     #step, target1
        DJN     bomber1, counter1

bomber2 MOV     bomb, >target2
        ADD     #step, target2
        DJN     bomber2, counter2

bomber3 MOV     bomb, >target3
        ADD     #step, target3
        DJN     bomber3, counter3

bomb    DAT     #0, #0

target0 DAT     #0, #0
target1 DAT     #0, #0
target2 DAT     #0, #0
target3 DAT     #0, #0

counter0 DAT    #0, #0
counter1 DAT    #0, #0
counter2 DAT    #0, #0
counter3 DAT    #0, #0

        END start
