;Name   Rato Replicante
;Author Rodrigo Setti
;Strat  MantÃÂ©m um processo ininterrupto de
;Strat  criaÃÂ§ÃÂ£o de cÃÂ³pias funcionais.

org inicio

vetores dat.f   $0,         $2981       ;vetores de copia
inicio  mov.i   }vetores,   >vetores    ;copia instruÃÂ§ÃÂ£o e incrementa vetores
        jmn.b   $inicio,    *vetores    ;loop de cÃÂ³pia -> enquanto nao encontrou um zero em B
        spl.b   <vetores,   {vetores    ;cria processo na cÃÂ³pia (ajuda a reestruturar ponteiro)
        add.x   #-31,       $-4         ;reestrutura ponteiros
        jmz.a   $inicio,    {vetores    ;loop do programa (ajuda a reestruturar ponteiro)
                                        ;ou se suicida se o programa estiver alterado
