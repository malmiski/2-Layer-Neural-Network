#|
Mohanad Almiski
|#



(defun shuffle (lis)
  "Shuffles a list.  Non-destructive.  O(length lis), so
pretty efficient.  Returns the shuffled version of the list."
  (let ((vec (apply #'vector lis)) bag (len (length lis)))
    (dotimes (x len)
      (let ((i (random (- len x))))
	(rotatef (svref vec i) (svref vec (- len x 1)))
	(push (svref vec (- len x 1)) bag)))
    bag))   ;; 65 s-expressions, by the way


(defparameter *verify* t)

(defun throw-error (str)
  (error (make-condition 'simple-error :format-control str)))

(defun verify-equal (funcname &rest matrices)
  ;; we presume they're rectangular -- else we're REALLY in trouble!
  (when *verify*
    (unless (and
	     (apply #'= (mapcar #'length matrices))
	     (apply #'= (mapcar #'length (mapcar #'first matrices))))
      (throw-error (format t "In ~s, matrix dimensions not equal: ~s"
			   funcname
			   (mapcar #'(lambda (mat) (list (length mat) 'by (length (first mat))))
				   matrices))))))

(defun verify-multiplicable (matrix1 matrix2)
  (when *verify*
    (if (/= (length (first matrix1)) (length matrix2))
	(throw-error (format t "In multiply, matrix dimensions not valid: ~s"
			     (list (list (length matrix1) 'by (length (first matrix1)))
				   (list (length matrix2) 'by (length (first matrix2)))))))))


;; Basic Operations

(defun map-m (function &rest matrices)
  "Maps function over elements in matrices, returning a new matrix"
  (apply #'verify-equal 'map-m  matrices)
  (apply #'mapcar #'(lambda (&rest vectors)       ;; for each matrix...
		      (apply #'mapcar #'(lambda (&rest elts)     ;; for each vector...
					  (apply function elts))
			     vectors)) 
	 matrices))   ;; pretty :-)

(defun transpose (matrix)
  "Transposes a matrix"
  (apply #'mapcar #'list matrix))  ;; cool, no?

(defun make-matrix (i j func)
  "Builds a matrix with i rows and j columns,
    with each element initialized by calling (func)"
  (map-m func (make-list i :initial-element (make-list j :initial-element nil))))

(defun make-random-matrix (i j val)
  "Builds a matrix with i rows and j columns,
    with each element initialized to a random
    floating-point number between -val and val"
  (make-matrix i j #'(lambda (x)
		       (declare (ignore x))  ;; quiets warnings about x not being used
		       (- (random (* 2.0 val)) val))))

(defun e (matrix i j)
  "Returns the element at row i and column j in matrix"
  (elt (elt matrix (1- i)) (1- j)))

(defun print-matrix (matrix)
  "Prints a matrix in a pleasing form, then returns matrix"
  (mapcar #'(lambda (vector) (format t "~%~{~8,4,,F~}" vector)) matrix) matrix)

;;; Matrix Multiplication

(defun multiply2 (matrix1 matrix2)
  "Multiplies matrix1 by matrix2 
    -- don't use this, use multiply instead"
  (verify-multiplicable matrix1 matrix2)
  (let ((tmatrix2 (transpose matrix2)))
    (mapcar #'(lambda (vector1)
		(mapcar #'(lambda (vector2)
			    (apply #'+ (mapcar #'* vector1 vector2))) tmatrix2))
	    matrix1)))  ;; pretty :-)

(defun multiply (matrix1 matrix2 &rest matrices)
  "Multiplies matrices together"
  (reduce #'multiply2 (cons matrix1 (cons matrix2 matrices))))

;;; Element-by-element operations

(defun add (matrix1 matrix2 &rest matrices)
  "Adds matrices together, returning a new matrix"
  (apply #'verify-equal 'add matrix1 matrix2 matrices)
  (apply #'map-m #'+ matrix1 matrix2 matrices))

(defun e-multiply (matrix1 matrix2 &rest matrices)
  "Multiplies corresponding elements in matrices together, 
        returning a new matrix"
  (apply #'verify-equal 'e-multiply matrix1 matrix2 matrices)
  (apply #'map-m #'* matrix1 matrix2 matrices))

(defun subtract (matrix1 matrix2 &rest matrices)
  "Subtracts matrices from the first matrix, returning a new matrix."
  (let ((all (cons matrix1 (cons matrix2 matrices))))
    (apply #'verify-equal 'subtract all)
    (apply #'map-m #'- all)))

(defun scalar-add (scalar matrix)
  "Adds scalar to each element in matrix, returning a new matrix"
  (map-m #'(lambda (elt) (+ scalar elt)) matrix))

(defun scalar-multiply (scalar matrix)
  "Multiplies each element in matrix by scalar, returning a new matrix"
  (map-m #'(lambda (elt) (* scalar elt)) matrix))

;;; This function could
;;; be done trivially with (scalar-add scalar (scalar-multiply -1 matrix))
(defun subtract-from-scalar (scalar matrix)
  "Subtracts each element in the matrix from scalar, returning a new matrix"
  (map-m #'(lambda (elt) (- scalar elt)) matrix))






(defun sigmoid (u)
  "Sigmoid function applied to the number u"
  (/ 1 (+ 1 (exp (- u)))))

(defun net-error (output correct-output)
  "Returns (as a scalar value) the error between the output and correct vectors"
  (let ((err (subtract output correct-output) ))
    (* 0.5 (first (first (multiply (transpose err) err))))))


(defun forward-propagate (datum v w)
  "Returns as a vector the output of the OUTPUT units when presented
the datum as input."
  (map-m #'sigmoid (multiply w (map-m #'sigmoid (multiply v (first datum))))))


(defun back-propagate (datum alpha v w)
  "Back-propagates a datum through the V and W matrices,
returning a list consisting of new, modified V and W matrices."
  (let* ((output (forward-propagate datum v w))
	 (one-minus-out (subtract-from-scalar 1 output))
	 (output-delta (e-multiply (e-multiply (subtract (second datum) output) output) one-minus-out)) 
	 (h (map-m #'sigmoid (multiply v (first datum)))) 
	 (one-minus-h (subtract-from-scalar 1 h))
	 (w-times-output-delta (multiply (transpose w) output-delta))
	 (h-delta (e-multiply (e-multiply h one-minus-h) w-times-output-delta)) 
	 (nablaV (scalar-multiply (- alpha 0) (multiply h-delta (transpose (first datum)))))
	 (nablaW (scalar-multiply (- alpha 0) (multiply output-delta (transpose h)))))
    ;; So after all those variable declarations finally return something
    (list (add nablaV v) (add nablaW w))))




(defun optionally-print (x option)
  "If option is t, then prints x, else doesn't print it.
In any case, returns x"
  ;;; perhaps this might be a useful function for you
  (if option (print x) x))


(defparameter *a-good-minimum-error* 1.0e-9)



(defun net-build (data num-hidden-units alpha initial-bounds max-iterations modulo &optional print-all-errors)
  ;; Initialize our V and W matrices
  (let ((v (make-random-matrix num-hidden-units (length (first (first data))) initial-bounds))
	(w (make-random-matrix (length (second (first data))) num-hidden-units initial-bounds))
	;; A place holder for the list of V and W that back-propagate returns
	newMatrices
	;; The output of forward-propagate
	output)
    ;; Iterate a maximum of max-iteration times
    ;; or until the error is not changing more
    ;; than a certain threshold
    (dotimes (x max-iterations)
      ;; Shuffle the training data around every iteration
      (let ((random-samples (shuffle data))
	    ;; Record error for logging and for breaking out
	    (worst-error 0)
	    (output-error 0)
	    (total-error 0))
	;; Go through each sample of the data and perform backprop
	;; for each data input
	(dotimes (index (length random-samples))
	  ;; Perform back-prop on the data at specified index
	  (setf newMatrices (back-propagate (elt random-samples index) alpha v w))
	  ;; Update our V and W matrices after backprop
	  (setf v (first newMatrices))
	  (setf w (second newMatrices))
	  ;; Record the new output and the corresponding error
	  (setf output (forward-propagate (elt random-samples index) v w))
	  (setf output-error (net-error output (second (elt random-samples index))))
	  (optionally-print output-error print-all-errors)
	  ;; Record the worst error out of all the samples
	  (if (> output-error worst-error)
	      (setf worst-error output-error))
	  (setf total-error (+ total-error output-error)))
	;; If this is the modulo'th iteration, then print out the results
	(if (= (mod (+ 1 x) modulo) 0)
	    (progn
	      (print worst-error)
	      (print (/ total-error (length data)))))
	;; If the error has reached a min
	(if (< worst-error *a-good-minimum-error*)
	    (return (list v w)))
	;; Reset errors for new run of training
	(setf worst-error 0)
	(setf total-error 0)))
    (list v w)))



  (defun simple-generalization (data num-hidden-units alpha initial-bounds max-iterations)
    (let* ((len (length data))
	   (converted-data (convert-data (scale-data data)))
	   (first-half (butlast converted-data (ceiling len 2)))
	   (second-half (last converted-data (ceiling len 2)))
	   ;; Build the network from the first half of the data
	   (vw (net-build first-half num-hidden-units alpha initial-bounds max-iterations max-iterations))
	   (sum-of-errors 0))
      (dotimes (x (length second-half))
	  (let* ((output (forward-propagate (elt second-half x) (first vw) (second vw)))
		 (neterror (net-error output (second (elt second-half x)))))
	         ;; Update the sum of errors
	         (setf sum-of-errors (+ sum-of-errors neterror))))
      (/ sum-of-errors (length second-half))))



  (defun k-fold-validation (data k num-hidden-units alpha initial-bounds max-iterations)
    (let ((total-error 0)
	  (total-samples 0)
	  (data (convert-data (scale-data data))))
      ;; Do this k times
      (dotimes (len k)
	;; Make x the length of a 1/k*len(data) chunk
      (let* ((x (ceiling (length data) k))  
	     (end (* len x))
	     (start (min (+ k (* len x)) (length data)))
	     ;; Train on the (k-1)/k data chunk
	     (network (net-build (append (subseq data 0 end) (subseq data start)) num-hidden-units alpha initial-bounds max-iterations max-iterations)))
	;; Get the 1/k test data chunk
	(dotimes (i (length (subseq data end start)))
	  (let* ((output (forward-propagate (elt data i) (first network) (second network)))
	         (err (net-error output (second (elt data i)))))
	    ;; Update total error with err from this output
	  (setf total-error (+ err total-error))))
	(setf total-samples (+ (length (subseq data end start)) total-samples))))
          ;; Divide total error by the number of samples
        (/ total-error total-samples)))




  (defun scale-list (lis)
    (let ((min (reduce #'min lis))
	  (max (reduce #'max lis)))
      (mapcar (lambda (elt) (+ 0.1 (* 0.8 (/ (- elt min) (- max min)))))
	      lis)))

  (defun scale-data (lis)
    (transpose (list (transpose (mapcar #'scale-list (transpose (mapcar #'first lis))))
		     (transpose (mapcar #'scale-list (transpose (mapcar #'second lis)))))))

  (defun convert-data (raw-data)
    (mapcar #'(lambda (datum)
		(mapcar #'(lambda (vec)
			    (mapcar #'list vec))
			(list (cons 0.5 (first datum))
			      (second datum))))
	    raw-data))

  (defun average (lis)
    (if (= (length lis) 0)
	0
	(/ (reduce #'+ lis) (length lis))))


  (defparameter *nand*
    '(((0.1 0.1) (0.9))
      ((0.9 0.1) (0.9))
      ((0.1 0.9) (0.9))
      ((0.9 0.9) (0.1))))


  (defparameter *xor*
    '(((0.1 0.1) (0.1))
      ((0.9 0.1) (0.9))
      ((0.1 0.9) (0.9))
      ((0.9 0.9) (0.1))))


