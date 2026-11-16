"""Common-sample scientific charts and a self-contained regional comparison."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

COUNTRIES = ['HU','PL','CZ','RO','SE']
NAMES = {'HU':'Hungary','PL':'Poland','CZ':'Czechia','RO':'Romania','SE':'Sweden'}
COMPONENTS = ['world','national_FX','other_FX','local']
LABELS = {'world':'World','national_FX':'National FX','other_FX':'Other FX','local':'Own local'}
COLORS = {'world':'#3979ad','national_FX':'#d38c27','other_FX':'#319680','local':'#926bad'}


def save_figure(fig, folder, name):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight',metadata={'Software':'Regional research synthesis'})
    with plt.rc_context({'svg.hashsalt':'regional-unhedged-comparison-v1','svg.fonttype':'none'}):
        fig.savefig(folder/(name+'.svg'),bbox_inches='tight',metadata={'Date':None,'Creator':'Regional research synthesis'})
    plt.show();plt.close(fig)
    return [name+'.png',name+'.svg']


def average_table(premiums, frames, availability):
    cols=[c+'_premium_pp' for c in COMPONENTS]+['fitted_mean_pp']
    result=premiums.groupby(['country','perspective','covariance'],sort=False)[cols].mean().reset_index()
    result['observed_mean_excess_pp']=[frames[(r.country,r.perspective)].rx_local.mean()*100 for r in result.itertuples()]
    result['mean_pricing_residual_pp']=result.observed_mean_excess_pp-result.fitted_mean_pp
    return result.merge(availability,on=['country','perspective','covariance'],validate='one_to_one')


def mean_comparison(means, folder, name, title):
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    values=means[[c+'_premium_pp' for c in COMPONENTS]].to_numpy()
    low=min(np.minimum(values,0).sum(axis=1).min(),means.observed_mean_excess_pp.min(),0)
    high=max(np.maximum(values,0).sum(axis=1).max(),means.observed_mean_excess_pp.max(),0)
    padding=max((high-low)*.06,.1)
    for row,cov in enumerate(['scalar','diagonal']):
        for col,currency in enumerate(['USD','EUR']):
            ax=axes[row,col];g=means.query('perspective == @currency and covariance == @cov').set_index('country').loc[COUNTRIES]
            x=np.arange(5);positive=np.zeros(5);negative=np.zeros(5)
            for component in COMPONENTS:
                values=g[component+'_premium_pp'].to_numpy();base=np.where(values>=0,positive,negative)
                ax.bar(x,values,bottom=base,width=.63,color=COLORS[component],alpha=.9)
                positive+=np.where(values>=0,values,0);negative+=np.where(values<0,values,0)
            ax.scatter(x,g.fitted_mean_pp,marker='D',s=35,c='#27303b',zorder=5)
            ax.scatter(x,g.observed_mean_excess_pp,marker='o',s=38,facecolors='none',edgecolors='#27303b',lw=1.1,zorder=5)
            labels=[NAMES[c]+(' †' if not bool(g.loc[c,'inference_ready']) else '') for c in COUNTRIES]
            ax.set(xticks=x,xticklabels=labels,title=f'{cov.capitalize()} BEKK · {currency} investor',ylabel='Monthly log percentage points')
            ax.axhline(0,color='#69717d',lw=.8);ax.grid(axis='x',visible=False)
            ax.set_ylim(low-padding,high+padding)
            ax.spines[['top','right']].set_visible(False)
    handles=[Patch(facecolor=COLORS[c],label=LABELS[c]) for c in COMPONENTS]+[
        Line2D([],[],marker='D',color='#27303b',ls='none',label='Fitted total'),
        Line2D([],[],marker='o',color='#27303b',markerfacecolor='none',ls='none',label='Observed mean excess return')]
    fig.legend(handles=handles,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.suptitle(title+'\nOctober 2014–September 2026 · unhedged equities · † ordinary inference withheld',fontsize=14)
    return save_figure(fig,folder,name)


def monthly_comparison(premiums, availability, currency, covariance, folder, name):
    data=premiums.query('perspective == @currency and covariance == @covariance').copy()
    data['currency_total_premium_pp']=data.national_FX_premium_pp+data.other_FX_premium_pp
    rows=[('world_premium_pp','World',COLORS['world']),('currency_total_premium_pp','Currency total',COLORS['national_FX']),('local_premium_pp','Own local',COLORS['local'])]
    fig,axes=plt.subplots(3,5,figsize=(18,9),sharex=True,sharey='row',layout='constrained')
    flags=availability.query('perspective == @currency and covariance == @covariance').set_index('country')
    for col,country in enumerate(COUNTRIES):
        g=data.loc[data.country.eq(country)].sort_values('month');dates=pd.PeriodIndex(g.month,freq='M').to_timestamp()
        assert len(g)==144
        for row,(field,label,color) in enumerate(rows):
            ax=axes[row,col];v=g[field].reset_index(drop=True)
            ax.plot(dates,v,color=color,lw=.7,alpha=.4)
            ax.plot(dates,v.rolling(12,min_periods=12).mean(),color=color,lw=1.7)
            ax.axhline(0,color='#69717d',lw=.7)
            for a,b in [('2020-01-01','2021-01-01'),('2022-01-01','2024-01-01')]:ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color='#7e8590',alpha=.07,lw=0)
            ax.spines[['top','right']].set_visible(False)
            if col==0:ax.set_ylabel(label+'\nMonthly log pp')
            if row==0:ax.set_title(NAMES[country]+(' †' if not bool(flags.loc[country,'inference_ready']) else ''),fontsize=12)
            if row==2:
                ax.set_xticks(pd.to_datetime(['2015-01-01','2020-01-01','2026-01-01']),['2015','2020','2026'])
    fig.suptitle(f'{currency} investor · {covariance} BEKK · common FX-only reference\nThin: monthly contribution · thick: trailing 12-month average · currency total = national FX + other FX · † inference withheld',fontsize=13)
    return save_figure(fig,folder,name)


def viewer_data(catalog_premiums, catalog_summary, targets_premiums, targets_summary, frames):
    labels={1:'S1 · constant prices',2:'S2 · varying world price',5:'FX · varying FX prices',3:'S3 · varying world + FX',4:'S4 · varying world + FX + local'}
    data={'countries':NAMES,'order':COUNTRIES,'components':COMPONENTS,'labels':LABELS,'months':pd.period_range('2014-10','2026-09',freq='M').astype(str).tolist(),
          'models':{str(s):labels[s] for s in [1,2,5,3,4]},'cases':{}}
    data['models'].update({'innovations':'FX · innovation-moment target','returns':'FX · centered-return target'})
    for (country,currency,cov,stage),g in catalog_premiums.groupby(['country','perspective','covariance','stage'],sort=False):
        r=catalog_summary.query('country == @country and perspective == @currency and covariance == @cov and stage == @stage').iloc[0]
        data['cases'][f'{stage}/{currency}/{cov}/{country}']={
            'series':np.round(g.sort_values('month')[[c+'_premium_pp' for c in COMPONENTS]].to_numpy(),6).tolist(),
            'ready':bool(r.inference_ready_QML),'reason':str(r.inference_reason_QML),
            'observed':round(frames[(country,currency)].rx_local.mean()*100,6),'AICrank':int(r.AIC_rank_all10),'BICrank':int(r.BIC_rank_all10)}
    for (country,currency,cov,kind),g in targets_premiums.query("target_kind != 'free'").groupby(['country','perspective','covariance','target_kind'],sort=False):
        r=targets_summary.query('country == @country and perspective == @currency and covariance == @cov and target_kind == @kind').iloc[0]
        data['cases'][f'{kind}/{currency}/{cov}/{country}']={
            'series':np.round(g.sort_values('month')[[c+'_premium_pp' for c in COMPONENTS]].to_numpy(),6).tolist(),
            'ready':bool(r.primary_inference_ready),'reason':str(r.primary_reason),'observed':round(frames[(country,currency)].rx_local.mean()*100,6),'AICrank':None,'BICrank':None}
    assert len(data['cases'])==140
    return data


def write_viewer(data, template, destination):
    template=Path(template).read_text();assert template.count('__REGIONAL_DATA__')==1
    payload=json.dumps(data,separators=(',',':'),ensure_ascii=True).replace('</','<\\/')
    text=template.replace('__REGIONAL_DATA__',payload)
    Path(destination).write_text(text)
    return len(text.encode())
