/******************************************************************/
/* Title:       CCSR FOR ICD-10-CM DIAGNOSIS MAPPING PROGRAM      */
/*                                                                */
/* Program:     DXCCSR_Mapping_Program_v2021-1.SAS                */
/*                                                                */
/* Description: This is the SAS mapping program to apply 		  */
/*			  the CCSR to the user’s ICD-10-CM-coded data. 	      */
/*			  The mapping program includes two options 	          */
/*			  for the file structure of CCSR output.              */
/*                                                                */
/*              There are two general sections to this program:   */
/*                                                                */
/*              1) The first section creates temporary SAS        */
/*                 informats using the DXCCSR CSV file.           */
/*                 These informats are used in step 2 to create   */
/*                 the CCSR variables in the output files.        */
/*                 output files.                                  */
/*              2) The second section loops through the diagnosis */
/*                 array in your SAS dataset and assigns          */
/*                 CCSR categories in the output files.           */
/*                                                                */
/*              Starting with v2021-1 of the CCSR, the SAS program*/
/*              and accompanying CSV file include a separate      */
/*              assignment of the default CCSR for inpatient data */
/*              (principal diagnosis) and outpatient data (first- */
/*              listed diagnosis). This feature requires that the */
/*              user identifies whether the input data is only    */
/*              inpatient (IP), only outpatient (OP), or a        */
/*              mixture of inpatient and outpatient data (IO).    */
/*              The v2021-1 (and later) SAS program cannot be used*/
/*              with the prior versions of the DXCCSR CSV file.   */
/*                                                                */
/*              The v2021-1 CSV file is compatible with ICD-10-CM */
/*              diagnosis codes from October 2015 through         */
/*              September 2021.                                   */
/*                                                                */
/* Output:      The program always outputs a vertical file.       */
/*              + Vertical file's layout: RECID DXCCSR DX_POSITION*/
/*                         DEFAULT_DXCCSR(Y, N, X, or Blank)      */
/*                         DXCCSR_VERSION                         */
/*                                                                */
/*		        The user can select to output two additional files*/
/*              + Horizontal file's layout: RECID DXCCSR_BBBNNN   */
/*                         DXCCSR_DEFAULT_DX1  DXCCSR_VERSION     */
/*              where BBB=3-letter body system abbreviation, 	  */
/*                     NNN=3-digit number                         */
/*              + Default file's layout: RECID DXCCSR_DEFAULT_DX1 */
/*                         DXCCSR_VERSION                         */
/******************************************************************/

/******************************************************************/
/*  Macro Variables that must be set to define the characteristics*/
/*  of your SAS discharge data. Change these values to match the  */
/*  number of diagnoses in your dataset. Change CORE to match the */
/*  name of your dataset. Set DXPREFIX to be the fixed part of    */
/*  the diagnosis variables. For example, if the diagnosis        */
/*  variables in your SAS dataset are I10_DX1, I10_DX2, ... then  */
/*  set DXPREFIX to I10_DX.                                       */
/******************************************************************/
%Let obs_ = MAX;                                                           *<=== Use smaller number for testing;

* FILE LOCATIONS ;
FILENAME INRAW1  'Location of CSV file, DXCCSR_v2021-1.CSV'   lRECL=3000;  *<=== USER MUST MODIFY;
LIBNAME  IN1     'Location of input discharge data';                       *<=== USER MUST MODIFY;
LIBNAME  OUT1    'Directory to store output file';                         *<=== USER MUST MODIFY;

* OPTIONAL FILES ;
* Build vertical file? 1=yes, 0=no;           %LET VERT=1;                          *<=== USER MUST MODIFY;
* Build default DXCCSR file? 1=yes, 0=no;     %LET DFLT=1;                          *<=== USER MUST MODIFY;
* Build horizontal file? 1=yes, 0=no;         %LET HORZ=1;                          *<=== USER MUST MODIFY;

* FILE NAMES ;
* Input SAS file member name;                 %LET CORE=INPUT_SAS_FILE;             *<=== USER MUST MODIFY;
* Output SAS file name, vertical;             %LET VERTFILE=OUTPUT_VERT_FILE_NAME;  *<=== USER MUST MODIFY; 
* Output SAS file name, horizontal;           %LET HORZFILE=OUTPUT_HORZ_FILE_NAME;  *<=== USER MUST MODIFY;
* Output SAS file name, default DXCCSR;       %LET DFLTFILE=OUTPUT_DFLT_FILE_NAME;  *<=== USER MUST MODIFY;

* INPUT DATA CHARACTERISTICS ;
* Database types, IP/OP/IO;                   %LET DBTYPE=IP;                       *<=== USER MUST MODIFY;
* Data element to identify IP/OP records;     %LET IOVAR=;                          *<=== USER MUST MODIFY;
* Data value to identify IP records;          %LET IOVALI=IP;                       *<=== USER MUST MODIFY;
* Data value to identify OP records;          %LET IOVALO=OP;                       *<=== USER MUST MODIFY;
* Linking key on input SAS data;              %LET RECID=KEY;                       *<=== USER MUST MODIFY; 
* Prefix of diagnosis variables;              %LET DXPREFIX=I10_DX;                 *<=== USER MUST MODIFY;
* Maximum number of DXs on any record;        %LET NUMDX=30;                        *<=== USER MUST MODIFY;
* Record specific DX Count variable, if not available leave it blank;            
                                              %LET NDXVAR=I10_NDX;                  *<=== USER MUST MODIFY;
											  
TITLE1 'CREATE ICD-10-CM DXCCSR TOOL CATEGORIES';
TITLE2 'USE WITH DISCHARGE ADMINISTRATIVE DATA THAT HAS ICD-10-CM CODES';

/******************* SECTION 1: CREATE INFORMATS ****************************/
/*  SAS Load the CCSR CSV file & convert into temporary SAS informats that  */
/*  will be used to assign the DXCCSR variables in the next step.           */
/****************************************************************************/
%LET CCSRN_ = 6;
DATA DXCCSR;
    LENGTH LABEL $1140;
    INFILE INRAW1 DSD DLM=',' END = EOF FIRSTOBS=2;
    INPUT
       START             : $CHAR7.
       I10Label          : $CHAR124.
       I10CCSRDeftIP     : $10.
       I10CCSRLabelDeftIP: $CHAR228.
       I10CCSRDeftOP     : $10.
       I10CCSRLabelDeftOP: $CHAR228.
       I10CCSR1          : $10.
       I10CCSRLabel1     : $CHAR228.
       I10CCSR2          : $10.
       I10CCSRLabel2     : $CHAR228.
       I10CCSR3          : $10.
       I10CCSRLabel3     : $CHAR228.
       I10CCSR4          : $10.
       I10CCSRLabel4     : $CHAR228.
       I10CCSR5          : $10.
       I10CCSRLabel5     : $CHAR228.
       I10CCSR6          : $10.
       I10CCSRLabel6     : $CHAR228.
    ;

   RETAIN HLO " " FMTNAME "$DXCCSR" TYPE  "J" ;
   
   LABEL = CATX("#", OF I10CCSR1-I10CCSR6, OF I10CCSRLabel1-I10CCSRLabel6) ;
   OUTPUT;

   IF EOF THEN DO ;
      START = " " ;
      LABEL = " " ;
      HLO   = "O";
      OUTPUT ;
   END ;
RUN;

PROC FORMAT LIB=WORK CNTLIN = DXCCSR;
RUN;

DATA DXCCSRL(KEEP=START LABEL FMTNAME TYPE HLO);
  SET DXCCSR(KEEP=I10CCSR:) END=EOF;

  RETAIN HLO " " FMTNAME "$DXCCSRL" TYPE  "J" ;

  ARRAY CCSRC(&CCSRN_) I10CCSR1-I10CCSR&CCSRN_;
  ARRAY CCSRL(&CCSRN_) I10CCSRLabel1-I10CCSRLabel&CCSRN_;  

  LENGTH START $6 LABEL $228;
  DO I=1 to &CCSRN_;
    IF NOT MISSING(CCSRC(I)) then do;
      START=CCSRC(I);
      LABEL=CCSRL(I);
      output;
    end;
  end;

  IF EOF THEN DO ;
     START = " " ;
     LABEL = " " ;
     HLO   = "O";
     OUTPUT;
  END;
run;

PROC SORT DATA=DXCCSRL NODUPKEY; 
  BY START; 
RUN;

PROC FORMAT LIB=WORK CNTLIN = DXCCSRL;
RUN;

DATA DXCCSRDEFT(KEEP=START LABEL FMTNAME TYPE HLO);
  SET DXCCSR(KEEP=START I10CCSRDeftIP rename=(I10CCSRDeftIP=LABEL)) END=EOF;

  RETAIN HLO " " FMTNAME "$DXCCSRDIP" TYPE  "J" ;
  output;
  IF EOF THEN DO ;
     START = " " ;
     LABEL = " " ;
     HLO   = "O";
     OUTPUT;
  END;
run;

PROC SORT DATA=DXCCSRDEFT NODUPKEY; BY START; RUN;

PROC FORMAT LIB=WORK CNTLIN = DXCCSRDEFT;
RUN;

DATA DXCCSRDEFT(KEEP=START LABEL FMTNAME TYPE HLO);
  SET DXCCSR(KEEP=START I10CCSRDeftOP rename=(I10CCSRDeftOP=LABEL)) END=EOF;

  RETAIN HLO " " FMTNAME "$DXCCSRDOP" TYPE  "J" ;
  output;
  IF EOF THEN DO ;
     START = " " ;
     LABEL = " " ;
     HLO   = "O";
     OUTPUT;
  END;
run;

PROC SORT DATA=DXCCSRDEFT NODUPKEY; BY START; RUN;

PROC FORMAT LIB=WORK CNTLIN = DXCCSRDEFT;
RUN;

/*********** SECTION 2: CREATE ICD-10-CM CCSR OUTPUT FILES ***********************/
/*  Create CCSR categories for ICD-10-CM using the SAS informats created         */
/*  in Section 1 and the diagnosis codes in your SAS dataset.                    */
/*  At most three separate output files are created plus a few intermediate files*/
/*  for the construction of the horizontal and default DXCCSR for DX1 file       */
/*********************************************************************************/  

%Macro dxccsr_vt(dbt=IP);
   DATA &dbt._out_vt (KEEP=&RECID DXCCSR DX_POSITION DEFAULT_DXCCSR DXCCSR_VERSION) 
        dxccsr_flags  (keep=&RECID flag_anydx flag_xxx flag_dx1)    
     ;
	 retain &RECID;
     LENGTH ICD10_Code $7 DXCCSR $6 DX_POSITION 3 DEFAULT_DXCCSR $1 DXCCSR_VERSION $6 flag_xxx $6;
     LABEL DEFAULT_DXCCSR = "Indication of default CCSR for principal/first-listed ICD-10-CM diagnosis"
	       DXCCSR = "CCSR category for ICD-10-CM diagnosis"
		   DX_POSITION = "Position of code in input diagnosis array"
		   DXCCSR_VERSION = "Version of CCSR for ICD-10-CM diagnoses"
		   ;

     SET &dbt._in;
	 by &RECID;
     retain DXCCSR_VERSION "2021.1";
     array A_DX(&NUMDX)       &DXPREFIX.1-&DXPREFIX.&NUMDX;

     %if &NDXVAR ne %then %let MAXNDX = &NDXVAR;
     %else %let MAXNDX=&NUMDX;
 
     flag_anydx =0; &dbt.flag_xxx=''; flag_dx1=0;
	 if not missing(&DXPREFIX.1) then flag_dx1=1;

     DO I=1 TO min(&MAXNDX, dim(A_dx));
       ICD10_CODE=A_DX(I);
       DX_POSITION=I;
       Default_DCCSR_Val =input(ICD10_CODE, $DXCCSRD&dbt..);

       if ICD10_CODE ^= '' then flag_anydx=1;

       CCSRString=INPUT(A_DX(I), $DXCCSR.); 
       if not missing(ICD10_CODE) and missing(CCSRString) then do;
	      ***invalid diagnosis found;
		  DXCCSR='InvlDX';
		  if I= 1 then DEFAULT_DXCCSR = 'X';
		  else DEFAULT_DXCCSR = '';
		  output &dbt._out_vt;
	   end;
       else if not missing(CCSRString) then do;
          ccsrn=(COUNTC(CCSRString,'#')+1)/2;
	      if ccsrn=1 then do;
             next_delim = findc(CCSRString,"#"); 
	         DXCCSR=substr(CCSRString,1, next_delim-1);

             if I=1 then do;
    		    if Default_DCCSR_Val = DXCCSR then DEFAULT_DXCCSR = 'Y';
	    		else if Default_DCCSR_Val =: 'XXX' then do;
            	  DEFAULT_DXCCSR = 'X';
				  flag_xxx=Default_DCCSR_Val;
				end;			
		    	else DEFAULT_DXCCSR = 'N';
			 end;
			 else DEFAULT_DXCCSR = '';
			 
	         output &dbt._out_vt;
	      end;
	      else do;
	       do j=1 to ccsrn;
             next_delim = findc(CCSRString,"#"); 
             DXCCSR=substr(CCSRString,1, next_delim-1);
             CCSRString=substr(CCSRString,next_delim+1);

             if I=1 then do;
			    if Default_DCCSR_Val = DXCCSR then DEFAULT_DXCCSR = 'Y';
			    else if Default_DCCSR_Val =: 'XXX' then do;
				  DEFAULT_DXCCSR = 'X';
  				  flag_xxx=Default_DCCSR_Val;
				end;  
			    else DEFAULT_DXCCSR = 'N';
			 end;
			 else DEFAULT_DXCCSR = '';
			 
	         output &dbt._out_vt;
	       end; 
	       do j=1 to ccsrn-1;
             next_delim = findc(CCSRString,"#"); 
	         CCSRString=substr(CCSRString,next_delim+1);
	       end; /*do j*/
	      end; /*else do*/
       end; /*not missing CCSString*/
     end; /*loop i*/
	 
	 output dxccsr_flags; 
run;
Title1 "Vertical file";
proc contents data=&dbt._out_vt varnum;
run;
Title2 "Sample print of vertical file";
proc print data=&dbt._out_vt (obs=10);
run;
%mend;

* =========================================================================== * 
* Count maximum number of DXCCSR values for each body system. 
* Please do not change this code. It is necessary to the program function.  
* =========================================================================== *;
%macro count_ccsr;
  DATA Body_sys;
    length body bnum $3 ;
    INFILE INRAW1 DSD DLM=',' END = EOF FIRSTOBS=2;
    INPUT
       START             : $CHAR7.
       I10Label          : $CHAR124.
       I10CCSRDeftIP     : $10.
       I10CCSRLabelDeftIP: $CHAR228.
       I10CCSRDeftOP     : $10.
       I10CCSRLabelDeftOP: $CHAR228.
       I10CCSR1          : $10.
       I10CCSRLabel1     : $CHAR228.
       I10CCSR2          : $10.
       I10CCSRLabel2     : $CHAR228.
       I10CCSR3          : $10.
       I10CCSRLabel3     : $CHAR228.
       I10CCSR4          : $10.
       I10CCSRLabel4     : $CHAR228.
       I10CCSR5          : $10.
       I10CCSRLabel5     : $CHAR228.
       I10CCSR6          : $10.
       I10CCSRLabel6     : $CHAR228.
    ;

    array ccsrs I10CCSR1-I10CCSR6;
    do over ccsrs;
      body=substr(ccsrs, 1, 3);
      bnum=substr(ccsrs, 4, 3);
      if body not in ('', 'XXX')  then output;
    end;
    keep body bnum;
   run;
   proc sort data=Body_sys; by body bnum ; run;
   data body_max;
     set body_sys;
     by body bnum;
     if last.body;
   run;
   %global mnbody;
   %global body_;
   proc sql noprint;
     select distinct body into :body_ separated by ' '
     from body_max
     ; 
   quit;
   data null;
     set body_max end=eof;
     if eof then call symput("mnbody", put(_N_, 2.)); 
   run; 

   %do i=1 %to &mnbody;
     %let b=%scan(&body_, &i);
     %global max&b. ;
   %end;  

   data null;
     set body_max end=eof;
     mbody="max" || body; 
     call symput(mbody, bnum); 
     if eof then call symput("mnbody", put(_N_, 2.)); 
   run; 

   %put verify macro definition:;
   %put mnbody=&mnbody;
   %do i=1 %to &mnbody;
     %let b=%scan(&body_, &i);
     %put max&b._ = &&&max&b;
   %end;  
%mend;

%macro dxccsr_hz(dbt=IP);
* =========================================================================== * 
* Create horizontal file layout using vertical file                           *
* =========================================================================== *;
Data DXCCSR_First(keep=&RECID DXCCSR) DXCCSR_second(keep=&RECID DXCCSR);
  set &dbt._out_vt;
  by &RECID;
  if DXCCSR ^= 'InvlDX';
  if DX_Position = 1 then output DXCCSR_First;
  else output DXCCSR_Second;
run;

proc sort data=DXCCSR_second nodupkey;
  by &RECID DXCCSR;
run;
proc sort data=DXCCSR_First;
  by &RECID DXCCSR;
run;

data DXCCSR;
  length DX_Position 3;
  merge DXCCSR_First(in=inp) DXCCSR_Second(in=ins);
  by &RECID DXCCSR;
  if inp and not ins then DX_Position = 1;
  else if ins and not inp then DX_Position = 3;
  else DX_Position = 2;
run;

proc transpose data=DXCCSR out=DXCCSR_Transposed(drop=_NAME_) prefix=DXCCSR_; 
  by &RECID;
  ID DXCCSR;
  Var DX_Position;
run; 

**** Some input records may not have any diagnosis codes or only invalid diagnosis codes 
     and not be represented in the vertical file.
     Ensure the horizontal output file has the same number of records as input file;
data &dbt._out_hz ;
  retain &RECID; 
  LENGTH
    %do i=1 %to &mnbody; 
      %let b=%scan(&body_, &i);
      DXCCSR_&b.001-DXCCSR_&b.&&max&b. 
    %end;
    3 ;
  Label
    %do i=1 %to &mnbody; 
      %let b=%scan(&body_, &i);
	  %do j=1 %to &&max&b.;
	     %if &j < 10 %then DXCCSR_&b.00&j = "Indication that at least one ICD-10-CM diagnosis on the record is included in CCSR &b.00&j" ;
	     %else %if &j < 100 %then DXCCSR_&b.0&j = "Indication that at least one ICD-10-CM diagnosis on the record is included in CCSR &b.0&j" ;
	     %else DXCCSR_&b.&j = "Indication that at least one ICD-10-CM diagnosis on the record is included in CCSR &b.&j" ;
	  %end;
    %end;
    ;
  merge DXCCSR_Transposed &dbt._dflt_file(in=ind);
  by &RECID;
  if not ind then abort;   ***Should never happen but safe guard, default file contains all records from input file;

  ***If no diagnoses are found on the record, set all DXCCSR_* values to 0;
  array a _numeric_;
  do over a;
    if a = . then a=0;
  end;
  drop DXCCSR_MBD015 DXCCSR_MBD016;
run;

Title1 "Horizontal file";
proc contents data=&dbt._out_hz varnum;
run;
Title2 "Sample print of horizontal file";
proc print data=&dbt._out_hz(obs=10);
run;
%mend;

%macro dxccsr_dflt(dbt=IP);
* ================================================================================= * 
* Create the default CCSR file with RECID & Default DXCCSR value using vertical file*
* ================================================================================= *;
data &dbt._dflt_file(keep=&RECID DXCCSR_DEFAULT_DX1);
  length DXCCSR_DEFAULT_DX1 $6;
  set &dbt._out_vt(keep=&RECID DX_POSITION DXCCSR DEFAULT_DXCCSR);
  by &RECID;
  
  if DX_POSITION = 1 and DEFAULT_DXCCSR in ('Y');
  DXCCSR_DEFAULT_DX1 = DXCCSR;
run;

**** Some input records may not have any diagnosis codes or only invalid diagnosis codes 
     and not be represented in the vertical file.
     Ensure the default DXCCSR output file has the same number of records as input file;
Data &dbt._dflt_file;
  retain &RECID;
  length DXCCSR_DEFAULT_DX1 DXCCSR_VERSION $6;
  label DXCCSR_DEFAULT_DX1 = "Default CCSR for principal/first-listed ICD-10-CM diagnosis"
        DXCCSR_VERSION = "Version of CCSR for ICD-10-CM diagnoses"
        ;
  merge dxccsr_flags(in=ini) 
		&dbt._dflt_file(in=ino) ;
  by &RECID;  
  retain DXCCSR_VERSION "2021.1";
  
  if not ini then abort;
  if not ino then do;
	if not flag_dx1 then DXCCSR_DEFAULT_DX1 = 'NoDX1';
    else if flag_xxx ^='' then DXCCSR_DEFAULT_DX1 = flag_xxx;
	else if flag_anydx then DXCCSR_DEFAULT_DX1 = 'InvlDX';
  end;
  drop flag_anydx flag_xxx flag_dx1;  
run;  

%mend;

%macro main;
   %count_ccsr;
   proc sort data=IN1.&CORE(obs=&obs_ keep=&RECID &NDXVAR &DXPREFIX.1-&DXPREFIX.&NUMDX %if &dbtype ne and &IOVAR ne %then &IOVAR;) out=&CORE.Skinny; by &RECID;  run;
   %if &DBTYPE = IO %then %do;
     %if &IOVAR NE %then %do;
	   data IP_in OP_in NIO;
	     set &CORE.Skinny;
		 by &RECID;
		 if &IOVAR = "&IOValI" then output IP_in;
		 else if &IOVAR = "&IOValO" then output OP_in;
		 else output NIO;
	   run;	 
		 
       %dxccsr_vt(dbt=IP);
       %dxccsr_dflt(dbt=IP);
       %dxccsr_vt(dbt=OP);
       %dxccsr_dflt(dbt=OP);
	   %if &vert = 1 %then %do;
	   data OUT1.&VERTFILE(SortedBy=&RECID);
	     set IP_out_vt OP_out_vt;
	     by &RECID;
	   run;  
	   %end;
       %if &dflt = 1 %then %do;
         data OUT1.&DFLTFILE(SortedBy=&RECID);
	       set IP_dflt_file OP_dflt_file NIO(keep=&RECID);
		   by &RECID;
	     run;	

         Title1 "Default DXCCSR file";
         proc contents data=OUT1.&DFLTFILE;
         run;
         Title2 "Sample print of default DXCCSR file";
         proc print data=OUT1.&DFLTFILE(obs=10);
           var &RECID DXCCSR_DEFAULT_DX1;
         run;
       %end;
       %if &horz = 1 %then %do; 
         %dxccsr_hz(dbt=IP);
         %dxccsr_hz(dbt=OP);
		 data out1.&HORZFILE(SortedBy=&RECID);
	     set IP_out_hz OP_out_hz NIO(keep=&RECID);
		 by &RECID;
	   run;	 

       %end;
     %end;
     %else ERROR 'IOVAR is not specified';	 
   %end;
   %else %if &DBTYPE = IP %then %do;
     data IP_in;
	     set &CORE.Skinny;
		 by &RECID;
     run;
	 
     %dxccsr_vt(dbt=IP);
     %if &vert = 1 %then %do;
	 data OUT1.&VERTFILE(SortedBy=&RECID);
	   set IP_out_vt;
	   by &RECID;
	 run;  
	 %end;
     %dxccsr_dflt(dbt=IP);
     %if &dflt = 1 %then %do;
       data OUT1.&DFLTFILE(SortedBy=&RECID);
	     set IP_dflt_file;
		 by &RECID;
	   run;	

       Title1 "Default DXCCSR file";
       proc contents data=OUT1.&DFLTFILE;
       run;
       Title2 "Sample print of default DXCCSR file";
       proc print data=OUT1.&DFLTFILE(obs=10);
         var &RECID DXCCSR_DEFAULT_DX1;
       run;
     %end;
     %if &horz = 1 %then %do; 
       %dxccsr_hz(dbt=IP);
   	   data out1.&HORZFILE(SortedBy=&RECID);
	     set IP_out_hz;
		 by &RECID;
	   run;	 
     %end;
   %end;
   %else %if &DBTYPE = OP %then %do;
     data OP_in;
	     set &CORE.Skinny;
		 by &RECID;
     run;
  
     %dxccsr_vt(dbt=OP);
     %if &vert = 1 %then %do; 
	 data OUT1.&VERTFILE(SortedBy=&RECID);
	   set OP_out_vt;
	   by &RECID;
	 run;  
	 %end;
     %dxccsr_dflt(dbt=OP);
     %if &dflt = 1 %then %do;
       data OUT1.&DFLTFILE(SortedBy=&RECID);
	     set OP_dflt_file;
		 by &RECID;
	   run;	

       Title1 "Default DXCCSR file";
       proc contents data=OUT1.&DFLTFILE;
       run;
       Title2 "Sample print of default DXCCSR file";
       proc print data=OUT1.&DFLTFILE(obs=10);
         var &RECID DXCCSR_DEFAULT_DX1;
       run;
     %end;
     %if &horz = 1 %then %do; 
       %dxccsr_hz(dbt=OP);
   	   data out1.&HORZFILE(SortedBy=&RECID);
	     set OP_out_hz;
		 by &RECID;
	   run;	 
     %end;
   %end;
   
%mend;
%main;

