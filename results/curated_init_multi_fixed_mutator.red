;redcode
;name curated_init_multi_fixed_mutator
;author ExplicitMutator
;strategy replicator using six processes
;assert 6

space1   equ 34
space2   equ 49
space3   equ (34+49)

start1   spl start2
         spl 35
p1       mov #6,p1
next     add #-31,new
         mov <p1,<new
new      spl @3331,7000
         jmz p1,p1
erase    mov 7995,<0
         dat @0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
start    spl wait1
         spl wait2
         jmp wait3
wait1    jmp 1
wait2    jmp 1
wait3    spl 1
         jmp start1

         spl 4010
         mov #6,0
         add #-31,2
         mov <-2,<4
         spl @3089,7000
         jmz -4,-4
         mov 7995,<4010
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
start2   spl 35
         mov #6,0
         add #-31,2
         mov <-2,<1
         spl @0,7000
         jmz -4,-4
         dat #0
         ADD 3371,<0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #7995
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #3119
         dat #0
         dat #0
         spl 40
         mov #6,0
         add #-31,2
         mov <-2,<1
         spl @0,7000
         jmz -4,-4
         mov 103,<7995
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         dat #0
         end start
