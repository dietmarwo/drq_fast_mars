
;name Skipper Improved v9
;author ChatGPT
;strategy Enhanced skipping bomber with dual bombing and efficient pointer advancing:
;          Uses post-increment indirect bombing for rapid spreading,
;          spawns two bombing processes for faster coverage,
;          advances pointer by step after two bombs to maintain spacing and efficiency,
;          replaces JMP loops with continuous SPL bombing for better task parallelism.
;          Changed JMP to SPL for continuous advancing and bombing processes and reduced chance of task starvation.
;          Replaced JMP start with SPL start to keep main loop alive and allow multitasking.

        ORG start

step    EQU 4                      ; Step size for spacing

start   MOV.AB  #step, pointer         ; Initialize pointer with step size
        SPL     bomb                  ; Spawn first bombing process
        SPL     bomb2                 ; Spawn second bombing process
        SPL     advance               ; Spawn advancing process for continuous pointer movement
        SPL     start                 ; Keep main loop alive with multitasking

bomb    MOV.AB  #0, }pointer         ; Bomb current target using post-increment indirect (A number)
        SPL     bomb                  ; Continuously spawn bombing process for continuous bombing
        DAT     #0, #0                ; Suicide to avoid task flooding if SPL queue full

bomb2   MOV.AB  #0, }pointer         ; Bomb next target, pointer post-incremented again
        SPL     bomb2                 ; Continuously spawn second bombing process
        DAT     #0, #0                ; Suicide to avoid task flooding if SPL queue full

advance ADD.AB  #step*2, pointer       ; Advance pointer by double step to maintain spacing after two bombs
        SPL     advance               ; Continuous advancing process
        DAT     #0, #0                ; Suicide to avoid task flooding if SPL queue full

pointer DAT     #0,     #0            ; Holds offset to bomb target (pointer)

        END start
