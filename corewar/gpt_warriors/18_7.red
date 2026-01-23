;Name Polen
;Author Rodrigo Setti
;Strat LanÃÂ§a vÃÂ¡rios "esporos" com processos
;Strat pela memÃÂ³ria, difÃÂ­ceis de serem localizados
;Strat e exterminados, porÃÂ©m sÃÂ£o inofensivos.

org 2

jmp.f   #0,     <-3

slt.b	$1,	#4
mov.i	$-2,	$973
spl.f   @-1,    }1
add.ab	#971,	$-2
jmp.f   $-4,    #0
