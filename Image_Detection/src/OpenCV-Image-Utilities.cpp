//
// File:	OpenCV-Image-Utilities.cpp
// Author:      Nicolas ROUGON
// Affiliation:	Institut Mines-Telecom | Telecom SudParis | ARTEMIS Department
// Date:	September 26, 2021
//
// Description:	OpenCV sample routine
// > Miscellaneous image utilities
//   - Overlay binary map onto graylevel / color image
//   - Find zero crossings of a float (single channel) 2D array

// Tutorial: docs.opencv.org/4.5.3/db/da5/tutorial_how_to_scan_images.html

#include "opencv2/core.hpp"                // OpenCV core routines
#include <iostream>

using namespace std;
using namespace cv;                        // OpenCV classes and routines

// Routines
#include "OpenCV-Image-Utilities.h"

//-----------------------------------------------------------------------------
// Overlay non-zero elements of an image onto a base image

void overlay_uchar_image(Mat in, Mat map, uchar color, Vec3b color3, Mat *out)
{
  int    i,j;
  int    nrows = in.rows;
  int    ncols = in.cols;
  int    nchannels_in = in.channels();
  int    nchannels_out = out->channels();
  uchar  *p_in, *p_map;

  switch(nchannels_out)
    {
    case 1:   // Graylevel overlay
      uchar  *p_out;

      for (j=0; j < nrows; j++) {
	p_in = in.ptr<uchar>(j);
	p_map = map.ptr<uchar>(j);
	p_out = out->ptr<uchar>(j);
	for (i=0; i < ncols; i++) {
	  if (p_map[i])
	    p_out[i] = color;
	  else
	    p_out[i] = p_in[i];
	}
      }
      break;
    case 3:   // BGR color overlay
      Vec3b  *p_out3;

      switch(nchannels_in) {
      case 1 :   // Graylevel image
	int    pixel;

	for (j=0; j < nrows; j++) {
	  p_in = in.ptr<uchar>(j);
	  p_map = map.ptr<uchar>(j);
	  p_out3 = out->ptr<Vec3b>(j);
	  for (i=0; i < ncols; i++) {
	    if (p_map[i]) {
	      p_out3[i][0] = color3[0];
	      p_out3[i][1] = color3[1];
	      p_out3[i][2] = color3[2];
	    }
	    else {
	      pixel = p_in[i];
	      p_out3[i][0] = pixel;
	      p_out3[i][1] = pixel;
	      p_out3[i][2] = pixel;
	    }
	  }
	}
	break;
      case 3:    // BGR image
	Vec3b  *p_in3;

	for (j=0; j < nrows; j++) {
	  p_in3 = in.ptr<Vec3b>(j);
	  p_map = map.ptr<uchar>(j);
	  p_out3 = out->ptr<Vec3b>(j);
	  for (i=0; i < ncols; i++) {
	    if (p_map[i]) {
	      p_out3[i][0] = color3[0];
	      p_out3[i][1] = color3[1];
	      p_out3[i][2] = color3[2];
	    }
	    else {
	      p_out3[i][0] = p_in3[i][0];
	      p_out3[i][1] = p_in3[i][1];
	      p_out3[i][2] = p_in3[i][2];
	    }
	  }
	}
	break;
      }
      break;
    }
}

//-----------------------------------------------------------------------------
// Concatenate horizontally two matrices with arbitrary height.
// Vertical padding is added below the matrix with smallest height.

void vpad_and_hconcat(Mat in1, Mat in2, Scalar color, Mat *out)
{
  Mat in_padded;
  Mat *the_mat1, *the_mat2;

  // Pad the matrix with smallest height
  the_mat1 = &in1;
  the_mat2 = &in2;

  if (in1.rows > in2.rows) {
    copyMakeBorder(in2, in_padded, 0, in1.rows - in2.rows, 0,0,
		   BORDER_CONSTANT, color);
    the_mat2 = &in_padded;
  } else {
    copyMakeBorder(in1, in_padded, 0, in2.rows - in1.rows, 0,0,
		   BORDER_CONSTANT, color);
    the_mat1 = &in_padded;
  }

  // Perform horizontal concatenation
  hconcat(*the_mat1, *the_mat2, *out);

  // Memory deallocation
  in_padded.release();
}

//-----------------------------------------------------------------------------
// Find zero crossings of a float (single channel) 2D array

void FindZeroCrossings(Mat in, Mat *out)
{
  int i, j;
  int nrows = in.rows;
  int ncols = in.cols;
  int w_index, n_index;

  uchar *p_out;
  float *p_prev, *p_cur;
  float w, n, cur, diff, vdiff;

  float threshold = 0.0;

  // Find Zero Crossings
  for (j=0; j < nrows; j++) {
    n_index = ((j==0) ? 1 : -1);
    p_prev = in.ptr<float>(j+n_index);
    p_cur = in.ptr<float>(j);
    p_out = out->ptr<uchar>(j);
    for (i=1; i < ncols; i++) {
      w_index = ((i==0) ? 1 : -1);
      cur = p_cur[i];
      w = p_cur[i+w_index];
      n = p_prev[i];
      diff = 0.0;
      if (((cur > 0) && (w < 0)) || ((cur < 0) && (w > 0)))
	diff = abs(cur - w);
      if (((cur > 0) && (n < 0)) || ((cur < 0) && (n > 0))) {
	vdiff = abs(cur - n);
	if (vdiff > diff)
	  diff = vdiff;
      }
      p_out[i] = (diff > threshold) ? 255 : 0;
    }
  }
}

//-----------------------------------------------------------------------------
