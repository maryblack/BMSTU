; =============================================================================
; Python function returning index of value in list.
; -----------------------------------------------------------------------------
; def finder(arr: list, len_arr: int, x: int) -> int:
;   left = 0
;   right = len_arr - 1
;
;   while left <= right:
;       mid_ind = (left + right) // 2
;       mid = arr[mid_ind]
;       if mid == x:
;           print('A')
;           return mid_ind
;       elif mid < x:
;           print('B')
;           left = mid_ind + 1
;       elif mid > x:
;           print('C')
;           right = mid_ind - 1
;
;   return -1

; Env is environment <x, n, res>
(declare-datatypes ()
    ((Env (mk-env
        (env-arr (Array Int Int))
        (env-len-arr Int)
        (env-x Int)
        (env-l Int)
        (env-r Int)
        (env-mid-ind Int)
        (env-s Int)
        ))))


(define-fun condLoop((e Env)) Bool
	(<= (env-l e) (env-r e)))

(define-fun statementA((e Env)) Env
	(let ((arr (env-arr e))
	      (len-arr (env-len-arr e))
	      (x (env-x e))
	      (l (env-l e))
	      (r (env-r e))
	      (mid-ind (env-mid-ind e))
	      (s (env-s e)))
	(mk-env arr len-arr x l r (div (+ l r) 2) s)))

(define-fun condEq((e Env)) Bool
	(= (select (env-arr e) (env-mid-ind e)) (env-x e)))

(define-fun statementB((e Env)) Env
	(let ((arr (env-arr e))
	      (len-arr (env-len-arr e))
	      (x (env-x e))
	      (l (env-l e))
	      (r (env-r e))
	      (mid-ind (env-mid-ind e))
	      (s (env-s e)))
	(mk-env arr len-arr x (+ r 1) r mid-ind mid-ind)))

(define-fun condLess((e Env)) Bool
	(< (select (env-arr e) (env-mid-ind e)) (env-x e)))

(define-fun statementC((e Env)) Env
	(let ((arr (env-arr e))
	      (len-arr (env-len-arr e))
	      (x (env-x e))
	      (l (env-l e))
	      (r (env-r e))
	      (mid-ind (env-mid-ind e))
	      (s (env-s e)))
	(mk-env arr len-arr x (+ mid-ind 1) r mid-ind s)))

(define-fun condGreater((e Env)) Bool
	(> (select (env-arr e) (env-mid-ind e)) (env-x e)))

(define-fun statementD((e Env)) Env
	(let ((arr (env-arr e))
	      (len-arr (env-len-arr e))
	      (x (env-x e))
	      (l (env-l e))
	      (r (env-r e))
	      (mid-ind (env-mid-ind e))
	      (s (env-s e)))
	(mk-env arr len-arr x l (- mid-ind 1) mid-ind s)))

; =============================================================================
; Definitions shared by all models
; -----------------------------------------------------------------------------

(declare-const ARR (Array Int Int))
(declare-const X Int)

; =============================================================================
; Model 0: A B
; -----------------------------------------------------------------------------

(push)
(define-const e1 Env (mk-env ARR 1 X 0 1 0 -1))
(assert (condLoop e1))
(define-const e2 Env (statementA e1))
(assert (condEq e2))
(define-const e3 Env (statementB e2))
(assert (not (condLoop e3)))

(echo "A B")
(check-sat)
(get-model)
(pop)

; =============================================================================
; Model 1: A C
; -----------------------------------------------------------------------------

(push)
(define-const e1 Env (mk-env ARR 1 X 0 1 0 -1))
(assert (condLoop e1))
(define-const e2 Env (statementA e1))
(assert (condLess e2))
(define-const e3 Env (statementC e2))
;(assert (not (condLoop e3)))
(assert (= (env-s e3) -1))

(echo "A C")
(check-sat)
(eval (select ARR 0))
(get-model)
(pop)


; =============================================================================
; Model 2: A C A B
; -----------------------------------------------------------------------------

(push)
(define-const e1 Env (mk-env ARR 3 X 0 2 0 -1))
(assert (condLoop e1))
(define-const e2 Env (statementA e1))
(assert (condLess e2))
(define-const e3 Env (statementC e2))
(assert (condLoop e3))
(define-const e4 Env (statementA e3))
(assert (condEq e4))
(define-const e5 Env (statementB e4))

(echo "A C A B")
(check-sat)
(eval (select ARR 0))
(eval (select ARR 1))
(eval (select ARR 2))
(get-model)
(pop)

; =============================================================================
; Model 3: A D A B
; -----------------------------------------------------------------------------

(push)
(define-const e1 Env (mk-env ARR 6 X 0 5 0 -1))
(assert (condLoop e1))
(define-const e2 Env (statementA e1))
(assert (condGreater e2))
(define-const e3 Env (statementD e2))
(assert (condLoop e3))
(define-const e4 Env (statementA e3))
(assert (condEq e4))
(define-const e5 Env (statementB e4))

(echo "A D A B")
(check-sat)
(eval (select ARR 0))
(eval (select ARR 1))
(eval (select ARR 2))
(eval (select ARR 3))
(eval (select ARR 4))
(eval (select ARR 5))
(get-model)
(pop)