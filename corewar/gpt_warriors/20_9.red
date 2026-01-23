;Name	ImpQuattro
;Author	Rodrigo Setti
;Strat AvanÃ§a pela memÃ³ria se autocopiando
;Strat (AnÃ¡logo ao IMP)

org 1

mov.a	#0,	>4	;ponteiros e danificador de codigo
mov.i	}-1,	>-1	;auto copia a frente
djn.b   $-1,    #4      ;loop com contador(4)
mov.ab	#4,	$3	;recarrega proximo loop
