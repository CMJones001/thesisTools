#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

''' Tools to aid in the creation of consistent figures in the tufte thesis. '''

marginWidth = 2
textWidth   = 4.2
fullWidth   = 6.2

def createAxes(nAxes, colWrap=4, axesHeight=4, aspect=1.0, **subplotsKwargs):
  ''' Create a figure and sub-axes, while specifing the size of the sub figures.

  This emulated the behaviour of the seaborn facet grid, where we give a size
  for the individual figures rather than the overall plot.

  The remaining axes are blanked using plt.axis('off')

  Parameters
  ----------
  nAxes: int
    Number of sub-axes to create
  colWrap: int
    How many columns in the figure, next axes will go into new rows.
  axesHeight: float
    Height of the axes, in inches by default
  aspect: float
    Width/Height ratio of the sub-axes

  Returns
  -------
  fig, [axes]
    The axes are returned in a flat array, [0,...,nAxes-1]
  '''

  nCols = min(nAxes, colWrap)
  nRows = np.ceil(nAxes/nCols).astype(np.int)
  nAxesBlank = nCols*nRows - nAxes

  # print(f'nAxes = {nAxes}, nBlank = {nAxesBlank}, nRows = {nRows}, nCols = {nCols}')
  # raise SystemExit

  axesWidth = axesHeight*aspect
  figWidth = nCols*axesWidth
  figHeight = nRows*axesHeight

  fig, axs = plt.subplots(ncols=nCols, nrows=nRows,
                          figsize=(figWidth, figHeight),
                          **subplotsKwargs)

  # Keep the axes returned in a 1d array
  if nRows > 1:
    axs = axs.flatten()

  # Blank the axis at the end of the list if we create more than specified
  for blankAxes in range(nAxesBlank):
    axs[-(blankAxes+1)].axis('off')

  return fig, axs

def hideAxisLables(ax):
  ''' Given an axis, remove the text and spacing used in the tick marks. '''
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  ax.axis('off')

def annotateAxis(ax, label, pos=None, **fontKwargs):
  ''' Place a label in the figure.

  This defaults to the top left, useful for labelling sub-figures.
  '''
  if pos is None:
    pos = (0.05, 0.9)

  ax.annotate(
    label,
    xy=pos,
    xycoords=('axes fraction'),
    horizontalalignment='left',
    verticalalignment='baseline',
    **fontKwargs,
  )

def setLabels(axs, ylabels=None, xlabels=None):
  ''' Given a set of axes, label those that are in the outermost positions.

  We use the in-built methods on the axes to create y-labels on the left most
  axes and add x-labels to the bottom row.
  '''
  for ax in axs:
    if ylabels is not None and ax.is_first_col():
      ax.set_ylabel(ylabels)
    if xlabels is not None and ax.is_last_row():
      ax.set_xlabel(xlabels)

def formatSI(number, precision=3, s=''):
  ''' Add SI metric prefix to the number and print to a given specification.

  If the magnitude of the number is not within the range 10^-24 to 10^24 then
  we simply return the number in SI notation.
  '''
  if number == 0: return f'{0:{s}.{precision}g} '
  # Round the number first, this tends catches the case when 999 rounds to 1000
  number = float(f'{number:.{precision}e}')
  baseThousand = np.floor(np.log(abs(number))/np.log(1e3))

  # Ensure that we have a single digits starting with a sign, so that we can
  # look them up in a dictionary.
  prefixKey = f'{int(baseThousand):+1d}'

  # Shift by multiples of 1000 so the number in the range (1, 1000]
  shortenedDigits = number/(1e3**baseThousand)

  prefixDict = {
    # '-8':'y', '-7':'z', '-6':'a', '-5':'f',
    '-4':'p', '-3':'n', '-2':'μ', '-1':'m', '+0':'' ,
    '+1':'k', '+2':'M', '+3':'G', '+4':'T',
    # '+5':'P', '+6':'E', '+7':'Z', '+8':'Y',
  }

  if prefixKey in prefixDict:
    return f'{shortenedDigits:{s}.{precision}g} {prefixDict[prefixKey]}'
  else:
    return f'{number:{s}.{precision}g} '

class Curve:
    def __init__(self, ax, func):
        self.ax = ax
        self.func = func
        self.n_points = 100

    def get_curve(self, min_=-1.0, max_=1.0):
        ''' Evaluate the function in the range (-1, 1)'''
        range_ = np.linspace(min_, max_, self.n_points)
        return range_, self.func(range_)

    def plot_curve(self, min_=-1, max_=1, **fmt):
        ''' Plot the curve on the given axis. '''
        range_, curve = self.get_curve(min_, max_)
        self.ax.plot(range_, curve, **fmt)

    def plot_on_curve(self, x, **fmt):
        ''' Add a point on the curve at a given value of φ. '''
        y_val = self.func(x)
        self.ax.scatter(x, y_val, **fmt)

    def plot_line(self, min_, max_, pos=0.5, label=None, **fmt):
        ''' Plot a straight line on the curve. '''
        y_min = self.func(min_)
        y_max = self.func(max_)

        self.ax.plot((min_, max_), (y_min, y_max), **fmt)

        # Quit here if there is no label to add
        if pos is None: Return
        y_label = (y_max - y_min)*pos + y_min
        x_label = (max_ - min_)*pos + min_

        if label is None: Return
        self.ax.annotate(
          label,
          (x_label, y_label), (-30, 30),
          textcoords="offset points",
          va='center_baseline', ha='right',
          arrowprops=dict(
            lw=0.8,
            arrowstyle='-|>',
            fc='k',
          )
          )


    def draw_connecting_hline(self, x, label=None, pos=0.5, **fmt):
        ''' Draw a line from the x-axis to the given point on the curve.

        Additionally add a label a fraction of the way along this line.
        '''
        min_y, max_y = self.ax.get_ylim()
        curve_y =  self.func(x)

        # Add the vertical line
        self.ax.vlines(x, min_y, curve_y, **fmt)

        # Reset the axis, as adding the vline may shift the axis
        self.ax.set_ylim(min_y, max_y)

        # Exit here if there is no label
        if label is None: return
        label_y = (curve_y - min_y)*pos + min_y
        self.ax.annotate(
            label, (x, label_y), (5, 0),
            textcoords="offset points",
            va='center', ha='left',
        )

    def draw_connecting_vline(self, x, label=None, pos=0.5, **fmt):
        ''' Draw a line from the y-axis to the given point on the curve.

        Additionally add a label a fraction of the way along this line.
        '''
        min_x, max_x = self.ax.get_xlim()
        curve_y =  self.func(x)

        # Add the horizontal line
        self.ax.hlines(curve_y, min_x, x, **fmt)

        # Reset the axis, as adding the vline may shift the axis
        self.ax.set_xlim(min_x, max_x)

        # Exit here if there is no label
        if label is None: return
        # label_y = (curve_y - min_y)*pos + min_y
        # self.ax.annotate(
        #     label, (x, label_y), (5, 0),
        #     textcoords="offset points",
        #     va='center', ha='left',
        # )

    def draw_curved_arrow(self, x_start, x_end, scale, color='k'):
        ''' Draw a curved arrow between the two points on the curve. '''
        y_start = self.func(x_start)
        y_end   = self.func(x_end)

        curvature_radius = 0.4*scale
        arrow = patches.FancyArrowPatch(
            (x_start, y_start),
            (x_end, y_end),
            connectionstyle=f"arc3,rad={curvature_radius}",
            arrowstyle='fancy,tail_width=0.5,head_width=4,head_length=8',
            zorder=20,
            shrinkA=10,
            shrinkB=10,
            ec=color,
            fc=color,
            lw=0.8,
        )

        self.ax.add_patch(arrow)


if __name__ == "__main__":
  # fig, ax = createAxes(10, 3, axesHeight=4)
  # plt.show()

  for i in np.logspace(-12, 12, num=13):
    i *= 1.234
    multiple = formatSI(i, precision=3)
    print(f'{i:>11,.6g} {multiple}L')
